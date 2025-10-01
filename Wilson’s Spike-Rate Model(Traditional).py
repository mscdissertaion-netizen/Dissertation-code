# -*- coding: utf-8 -*-
"""
@author: Chetan Mathias
"""

import matplotlib.pyplot as plt
import numpy as np

# Time settings
time_step = 0
time_interval = 10000  # ms
dt = 0.25  # ms
time = []

# Model parameters
tau_E = 20       # Excitatory time constant (ms)
tau_I = 11       # Inhibitory time constant (ms)
tau_H = 900      # Adaptation time constant (ms)
E_V = 10         # External input to vertical neuron (left eye)
E_H = 10         # External input to horizontal neuron (right eye)
I_wt_mono = 0.45 # Inhibitory weight for monocular stage
I_wt_bin = 1.53 * I_wt_mono  # Stronger inhibition for binocular stage
H_wt = 0.47      # Adaptation weight
mon_to_bin_gain = 0.75  # Gain from monocular to binocular stage

# Initialise neuron activity lists
# Monocular stage - Left eye (vertical)
V_left = [30]     # Vertical-selective neuron (left eye)
I_Vleft = [10]    # Inhibitory neuron
H_Vleft = [15]    # Adaptation

# Monocular stage - Right eye (horizontal)
H_right = [10]    # Horizontal-selective neuron (right eye)
I_Hright = [8]    # Inhibitory neuron
H_Hright = [3]    # Adaptation

# Binocular stage
V_bin = [20]      # Binocular vertical-selective
H_bin = [15]      # Binocular horizontal-selective
I_Vbin = [12]     # Inhibitory neurons
I_Hbin = [10]     # Inhibitory neurons
H_Vbin = [8]      # Adaptation
H_Hbin = [5]      # Adaptation

# RK4 solver for 3D system
def RKThreeD(tau_E, tau_I, tau_H, p, E_fr, I_fr, H_fr, E_fcn, I_fcn, H_fcn, dt):
    k1E = dt * E_fcn(tau_E, p, H_fr, E_fr)
    k1I = dt * I_fcn(tau_I, I_fr, E_fr)
    k1H = dt * H_fcn(tau_H, H_fr, E_fr)

    k2E = dt * E_fcn(tau_E, p, H_fr + k1H/2, E_fr + k1E/2)
    k2I = dt * I_fcn(tau_I, I_fr + k1I/2, E_fr + k1E/2)
    k2H = dt * H_fcn(tau_H, H_fr + k1H/2, E_fr + k1E/2)

    k3E = dt * E_fcn(tau_E, p, H_fr + k2H/2, E_fr + k2E/2)
    k3I = dt * I_fcn(tau_I, I_fr + k2I/2, E_fr + k2E/2)
    k3H = dt * H_fcn(tau_H, H_fr + k2H/2, E_fr + k2E/2)

    k4E = dt * E_fcn(tau_E, p, H_fr + k3H, E_fr + k3E)
    k4I = dt * I_fcn(tau_I, I_fr + k3I, E_fr + k3E)
    k4H = dt * H_fcn(tau_H, H_fr + k3H, E_fr + k3E)

    E_fr = E_fr + (k1E + 2*k2E + 2*k3E + k4E)/6
    I_fr = I_fr + (k1I + 2*k2I + 2*k3I + k4I)/6
    H_fr = H_fr + (k1H + 2*k2H + 2*k3H + k4H)/6

    return E_fr, I_fr, H_fr

# Differential equations for monocular stage
def E_dot_mono(tau, P, H, E):
    return (1.0/tau) * (-E + 100 * P**2 / ((10 + H)**2 + P**2))

def I_dot_mono(tau, I, E):
    return (1.0/tau) * (E - I)

def H_dot_mono(tau, H, E):
    return (1.0/tau) * (H_wt * E - H)

# Differential equations for binocular stage
def E_dot_bin(tau, P, H, E):
    return (1.0/tau) * (-E + 100 * P**2 / ((10 + H)**2 + P**2))

# Simulation loop
for t in np.arange(0, time_interval, dt):
    time.append(t)

    # Monocular Stage 
    # Left eye (vertical)
    P_Vleft = E_V - I_wt_mono * I_Hright[time_step]
    P_Vleft = max(0, P_Vleft)
    
    Vleft, I_Vleft_new, H_Vleft_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Vleft,
        V_left[time_step], I_Vleft[time_step], H_Vleft[time_step],
        E_dot_mono, I_dot_mono, H_dot_mono, dt
    )
    
    # Right eye (horizontal)
    P_Hright = E_H - I_wt_mono * I_Vleft[time_step]
    P_Hright = max(0, P_Hright)
    
    Hright, I_Hright_new, H_Hright_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Hright,
        H_right[time_step], I_Hright[time_step], H_Hright[time_step],
        E_dot_mono, I_dot_mono, H_dot_mono, dt
    )
    
    # Update monocular activities
    V_left.append(Vleft)
    I_Vleft.append(I_Vleft_new)
    H_Vleft.append(H_Vleft_new)
    H_right.append(Hright)
    I_Hright.append(I_Hright_new)
    H_Hright.append(H_Hright_new)
    
    #  Binocular Stage 
    # For vertical binocular neuron:
    P_Vbin = mon_to_bin_gain * Vleft - I_wt_bin * I_Hbin[time_step]
    P_Vbin = max(0, P_Vbin)
    
    # For horizontal binocular neuron:
    P_Hbin = mon_to_bin_gain * Hright - I_wt_bin * I_Vbin[time_step]
    P_Hbin = max(0, P_Hbin)
    
    # Solve for binocular neurons
    Vbin, I_Vbin_new, H_Vbin_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Vbin,
        V_bin[time_step], I_Vbin[time_step], H_Vbin[time_step],
        E_dot_bin, I_dot_mono, H_dot_mono, dt
    )
    
    Hbin, I_Hbin_new, H_Hbin_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Hbin,
        H_bin[time_step], I_Hbin[time_step], H_Hbin[time_step],
        E_dot_bin, I_dot_mono, H_dot_mono, dt
    )
    
    # Update binocular activities
    V_bin.append(Vbin)
    I_Vbin.append(I_Vbin_new)
    H_Vbin.append(H_Vbin_new)
    H_bin.append(Hbin)
    I_Hbin.append(I_Hbin_new)
    H_Hbin.append(H_Hbin_new)
    
    time_step += 1

# Plotting
plt.figure(figsize=(14, 10))

# Set global font sizes
plt.rc('xtick', labelsize=22)    # X-axis tick label size
plt.rc('ytick', labelsize=22)    # Y-axis tick label size
plt.rc('axes', labelsize=22)     # Axis label size
plt.rc('axes', titlesize=22)     # Subplot title size
plt.rc('legend', fontsize=20)    # Legend font size

# Left monocular stage
plt.subplot(3, 1, 1)
plt.plot(time, V_left[1:], 'r', linewidth=2, label='Left Monocular (Vertical)')
plt.axvspan(0, 150, color='gray', alpha=0.2, label='Initial joint activation')
plt.title('Left Monocular Stage', fontsize=22)
plt.ylabel('Firing Rate', fontsize=22)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 10000)
plt.ylim(0, 60)

# Right monocular stage
plt.subplot(3, 1, 2)
plt.plot(time, H_right[1:], 'b', linewidth=2, label='Right Monocular (Horizontal)')
plt.axvspan(0, 150, color='gray', alpha=0.2)
plt.title('Right Monocular Stage', fontsize=22)
plt.ylabel('Firing Rate', fontsize=22)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 10000)
plt.ylim(0, 60)

# Binocular stage
plt.subplot(3, 1, 3)
plt.plot(time, V_bin[1:], 'r', linewidth=2, label='Binocular Vertical')
plt.plot(time, H_bin[1:], 'b', linewidth=2, label='Binocular Horizontal')
plt.title('Higher Binocular Stage', fontsize=22)
plt.xlabel('Time (ms)', fontsize=22)
plt.ylabel('Firing Rate', fontsize=22)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 10000)
plt.ylim(0, 85)

plt.tight_layout()
# plt.suptitle('Hierarchical Rivalry Dynamics', fontsize=20, y=1.02)

plt.show()
