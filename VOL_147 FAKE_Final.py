# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 21:25:34 2025

@author: Chetan Mathias
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, savgol_filter
from scipy.optimize import least_squares
from sklearn.preprocessing import minmax_scale
from skimage.transform import resize


# Load 4D fMRI data
func_path = r"C:\Users\mathi\OneDrive\Desktop\Final Project\Travelling waves data\VOL147_fMRI_binRivalry_FAKE.nii"
func_img = sitk.ReadImage(func_path)
func_data = sitk.GetArrayFromImage(func_img)  # Shape: (T, Z, Y, X)
print("Functional data shape:", func_data.shape)


# Load the mask
mask_path = r"C:\Users\mathi\OneDrive\Desktop\Final Project\Travelling waves data\lh-benson_periphery_V1_restrictedAngle_functional_thr_fake_2.nii"
mask_img = sitk.ReadImage(mask_path)
mask_data = sitk.GetArrayFromImage(mask_img)
print("Original mask shape:", mask_data.shape)


# Adjust mask dimensions if needed
if mask_data.ndim == 2:
    mask_data = np.expand_dims(mask_data, axis=0)

print("Adjusted mask shape:", mask_data.shape)
assert mask_data.shape == func_data.shape[1:], "Mask and fMRI data dimensions still do not match!"


# Extract voxel coordinates from mask
mask_coords = np.where(mask_data > 0)  # (Z, Y, X)
n_voxels = len(mask_coords[0])
print(f"Number of voxels in mask: {n_voxels}")


# Parameters
baseline_start = 155
Run_length = 3150
n_trials = 35
trial_length = 90  #  (9s)


# Extract timecourses for voxels in mask
voxel_timecourses = np.zeros((n_voxels, func_data.shape[0]))
for ivoxel, (z, y, x) in enumerate(zip(*mask_coords)):
    voxel_timecourses[ivoxel, :] = func_data[:, z, y, x]


# Trial averaging
voxel_trial_avg = np.zeros((n_voxels, trial_length))

for ivoxel in range(n_voxels):
    # Extract voxel timecourse (baseline period)
    voxel_tc = voxel_timecourses[ivoxel, baseline_start:]
    voxel_tc = voxel_tc[:Run_length]
    voxel_tc = detrend(voxel_tc)

    # Reshape into (time x trials)
    trial_matrix = voxel_tc.reshape((trial_length, n_trials), order='F')

    # Remove bad trials if any
    bad_trials = []  
    good_trials = [t for t in range(n_trials) if t not in bad_trials]
    trial_matrix = trial_matrix[:, good_trials]

    # Average across trials
    mean_trial_tc = np.mean(trial_matrix, axis=1)

    # Apply Savitzky-Golay filter
    mean_trial_tc = savgol_filter(mean_trial_tc, window_length=11, polyorder=1)
    voxel_trial_avg[ivoxel, :] = mean_trial_tc


# Normalise responses
voxel_trial_norm = voxel_trial_avg / np.max(np.abs(voxel_trial_avg))

# Time vector
time_vec = np.arange(0.1, 9.1, 0.1)  # 0.1 to 9.0
time_vec = time_vec - 2  # baseline correction

# Plot all voxel responses
plt.figure()

plt.plot(time_vec, voxel_trial_norm.T)
plt.xlabel('Time (s) From Stimulus Onset')
plt.ylabel('Normalised BOLD Response')
plt.ylim([-0.8, 1.2])
plt.title("Voxel Responses for the 'Fake' Trial")
plt.show()

# Sinusoidal fitting
def sinusoid(p, t):
    return p[0] * np.sin(p[1] * t + p[2]) + p[3]

phase_delays = []
response_amplitudes = []

for ivoxel in range(n_voxels):
    initial_guess = [1, 1.0, 3.85, 0]
    bounds = ([0, 0.9, -np.inf, -np.inf], [np.inf, 1.1, np.inf, np.inf])

    res = least_squares(
        lambda p: sinusoid(p, time_vec) - voxel_trial_norm[ivoxel, :],
        initial_guess,
        bounds=bounds
    )

    p_fit = res.x
    phase_delays.append(p_fit[2])
    response_amplitudes.append(p_fit[0])


# Create phase and amplitude maps
phase_map = np.zeros_like(mask_data, dtype=float)
amplitude_map = np.zeros_like(mask_data, dtype=float)
z_coords, y_coords, x_coords = mask_coords

for z, y, x, delay, amp in zip(z_coords, y_coords, x_coords, phase_delays, response_amplitudes):
    phase_map[int(z), int(y), int(x)] = delay
    amplitude_map[int(z), int(y), int(x)] = amp


# Custom colormap
hexMap = [
    'C0C0C0', '808080', '404040', '000000', 'FF99CC', '9999FF', '3333FF',
    '000099', '3399FF', '0066CC', '99CCFF', '66B2FF', '66FFFF', '006633',
    '00CC66', '66FF66', '00FF00', '009900', 'FFFF99', 'FFFF00', 'CCCC00',
    'FFb266', 'CC6600', '994C00', 'FF9999', 'FF0000', 'CC0000', '990000'
]

colormap_rgb = np.zeros((len(hexMap), 3))
for i, color in enumerate(hexMap):
    r = int(color[0:2], 16) / 255.0
    g = int(color[2:4], 16) / 255.0
    b = int(color[4:6], 16) / 255.0
    colormap_rgb[i] = [r, g, b]

# Scale to 256 levels
numLevels = 256
colormap_rgb = resize(colormap_rgb, (numLevels, 3), anti_aliasing=True)
colormap_rgb = minmax_scale(colormap_rgb, feature_range=(0, 1))


# Show middle slice of phase map
plt.figure()
plt.imshow(
    phase_map[phase_map.shape[0] // 2, :, :],
    cmap=plt.cm.colors.ListedColormap(colormap_rgb)
)
plt.colorbar()
plt.title('Phase Delay Map')
plt.show()

# Save phase & amplitude maps as NIfTI
size_4d = list(func_img.GetSize())  # (X, Y, Z, T)
index = [0, 0, 0, 0]
size = [size_4d[0], size_4d[1], size_4d[2], 0]  # drop time dimension
func_ref = sitk.Extract(func_img, size, index)

# Save phase map
phase_img = sitk.GetImageFromArray(phase_map)
phase_img.CopyInformation(func_ref)
sitk.WriteImage(phase_img, r"C:\Users\mathi\OneDrive\Desktop\Final Project\phase_delay\phase_delay_map_fake_lh.nii.gz")

# Save amplitude map
amp_img = sitk.GetImageFromArray(amplitude_map)
amp_img.CopyInformation(func_ref)
sitk.WriteImage(amp_img, 'amplitude_map.nii.gz')

print("Saved 'phase_delay_map.nii.gz' and 'amplitude_map.nii.gz', aligned to functional data.")

