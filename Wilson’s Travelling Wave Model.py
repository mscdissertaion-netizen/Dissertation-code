# -*- coding: utf-8 -*-
"""
@author: Chetan Mathias
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


stimulus_type = 'radial'  # options: 'radial', 'concentric', 'spiral'


# Parameters
N = 42
dx = 0.05  # cm (0.5 mm cortical spacing)
dt = 0.25  # ms
T_total = 2000  # ms
steps = int(T_total / dt)

# Time constants
tau_E = 20
tau_I = 11
tau_H = 900

# Stimulus strengths
E_T = 30
E_S = 24

# Spatial constants
j = 0.1  # cm inhibition spread
a = j  # inhibition kernel width

# Facilitation parameters
if stimulus_type == 'radial':
    g = 0.0
    a_facil = 2 * a
elif stimulus_type == 'concentric':
    g = 0.04
    a_facil = 2 * a
elif stimulus_type == 'spiral':
    g = 0.02
    a_facil = a

# Kernel definitions 
def exp5_kernel_ring(N, a, dx):
    center = N // 2
    indices = np.arange(N)
    dist = np.minimum(np.abs(indices - center), N - np.abs(indices - center)) * dx
    return np.exp(-dist**5 / a**5)

# Inhibition kernel
W_inhib = exp5_kernel_ring(N, a, dx)
W_inhib *= 0.27
W_inhib /= W_inhib.sum()

# Facilitation kernel
if g > 0:
    W_facil = exp5_kernel_ring(N, a_facil, dx)
    W_facil *= g
    W_facil /= W_facil.sum()
else:
    W_facil = np.zeros(N)

#  Derivative functions 
def dIT(IT, T): return (-IT + T) / tau_I
def dIS(IS, S): return (-IS + S) / tau_I
def dHT(HT, T): return (-HT + 2*T) / tau_H
def dHS(HS, S): return (-HS + 2*S) / tau_H
def dT(T, Pplus, H): return (-T + 100 * Pplus**2 / ((10 + H)**2 + Pplus**2)) / tau_E
def dS(S, Pplus, H): return (-S + 100 * Pplus**2 / ((10 + H)**2 + Pplus**2)) / tau_E

# Initialise arrays 
T = np.zeros((steps, N))
S = np.zeros((steps, N))
IT = np.zeros((steps, N))
IS = np.zeros((steps, N))
HT = np.zeros((steps, N))
HS = np.zeros((steps, N))

# Initial conditions
S[0, :] = 50
T[0, N==0] = 60

# Wave tracking
wave_angles = []
wave_times = []

# Simulation loop 
for t in range(1, steps):
    IT[t] = IT[t-1] + dt * dIT(IT[t-1], T[t-1])
    IS[t] = IS[t-1] + dt * dIS(IS[t-1], S[t-1])

    I_T = convolve(IS[t], W_inhib, mode='same')
    I_S = convolve(IT[t], W_inhib, mode='same')

    if g > 0:
        F_T = convolve(T[t-1], W_facil, mode='same')
        F_S = convolve(S[t-1], W_facil, mode='same')
    else:
        F_T = np.zeros(N)
        F_S = np.zeros(N)

    P_T = E_T - I_T + F_T
    P_S = E_S - I_S + F_S
    P_Tplus = np.maximum(P_T, 0)
    P_Splus = np.maximum(P_S, 0)

    T[t] = T[t-1] + dt * dT(T[t-1], P_Tplus, HT[t-1])
    S[t] = S[t-1] + dt * dS(S[t-1], P_Splus, HS[t-1])
    HT[t] = HT[t-1] + dt * dHT(HT[t-1], T[t])
    HS[t] = HS[t-1] + dt * dHS(HS[t-1], S[t])

    if np.max(T[t]) > 30:
        peak_idx = np.argmax(T[t])
        angle = 2 * np.pi * peak_idx / N
        wave_angles.append(angle)
        wave_times.append(t * dt / 1000.0)

#  Analyse wave propagation 
wave_times = np.array(wave_times)
wave_angles = np.array(wave_angles)
unwrapped_angles = np.unwrap(wave_angles)
unwrapped_pos = unwrapped_angles * (N * dx) / (2 * np.pi)  # cm

# Linear fit
A = np.vstack([wave_times, np.ones_like(wave_times)]).T
slope, intercept = np.linalg.lstsq(A, unwrapped_pos, rcond=None)[0]
y_pred = slope * wave_times + intercept
r_squared = 1 - np.sum((unwrapped_pos - y_pred)**2) / np.sum((unwrapped_pos - np.mean(unwrapped_pos))**2)
speed_deg = slope / 0.6

#  Output 
print(f"\nStimulus type: {stimulus_type}")
print(f"Estimated wave speed: {slope:.2f} cm/s (cortical)")
print(f"Equivalent to: {speed_deg:.2f} °/s (visual angle)")
print(f"R² of linear fit: {r_squared:.3f}")

# Plot: Cortical Distance vs Time 
plt.figure(figsize=(10, 4))
plt.plot(wave_times, unwrapped_pos, label='Wavefront position')
plt.plot(wave_times, y_pred, 'r--', label=f'Fit: {slope:.2f} cm/s')
plt.xlabel('Time (s)', fontsize=22)
plt.ylabel('Cortical distance (cm)', fontsize=22)
plt.title(f'Cortical Distance vs Time — {stimulus_type.capitalize()} Stimulus', fontsize=22)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.imshow(T[:2000], aspect='auto', cmap='hot', origin='lower')
plt.xlabel('Neuron index',fontsize =22)
plt.ylabel('Time step', fontsize =22)
plt.title('Wave propagation around the ring')
plt.colorbar(label='Activity')

plt.show()
