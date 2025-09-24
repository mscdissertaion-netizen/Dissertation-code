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
E_V = 10         # External input to vertical neuron
E_H = 10         # External input to horizontal neuron
I_wt_mono = 0.45
I_wt_bin = 1.53 * I_wt_mono
H_wt = 0.47
mon_to_bin_gain = 0.75

# F&S stimulus parameters
flicker_freq = 18.0  # Hz
swap_freq = 1.5      # Hz
flicker_period = 1000 / flicker_freq
swap_period = 1000 / swap_freq

# Initialize neuron activity lists
V_left = [30]
I_Vleft = [10]
H_Vleft = [15]

H_right = [10]
I_Hright = [8]
H_Hright = [3]

V_bin = [20]
H_bin = [15]
I_Vbin = [12]
I_Hbin = [10]
H_Vbin = [8]
H_Hbin = [5]

# RK4 Solver
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

    E_fr += (k1E + 2*k2E + 2*k3E + k4E)/6
    I_fr += (k1I + 2*k2I + 2*k3I + k4I)/6
    H_fr += (k1H + 2*k2H + 2*k3H + k4H)/6

    return E_fr, I_fr, H_fr

# Differential equations
def E_dot(tau, P, H, E):
    return (1.0/tau) * (-E + 100 * P**2 / ((10 + H)**2 + P**2))

def I_dot(tau, I, E):
    return (1.0/tau) * (E - I)

def H_dot(tau, H, E):
    return (1.0/tau) * (H_wt * E - H)

# Simulation loop
for t in np.arange(0, time_interval, dt):
    time.append(t)

    flicker = 1 if (t % flicker_period) < (flicker_period / 2) else 0
    swap_phase = int(t / swap_period) % 2

    if swap_phase == 0:
        current_E_V = E_V * flicker
        current_E_H = E_H * flicker
    else:
        current_E_H = E_V * flicker
        current_E_V = E_H * flicker

    # Monocular stage
    P_Vleft = max(0, current_E_V - I_wt_mono * I_Hright[time_step])
    Vleft, I_Vleft_new, H_Vleft_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Vleft,
        V_left[time_step], I_Vleft[time_step], H_Vleft[time_step],
        E_dot, I_dot, H_dot, dt
    )

    P_Hright = max(0, current_E_H - I_wt_mono * I_Vleft[time_step])
    Hright, I_Hright_new, H_Hright_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Hright,
        H_right[time_step], I_Hright[time_step], H_Hright[time_step],
        E_dot, I_dot, H_dot, dt
    )

    V_left.append(Vleft)
    I_Vleft.append(I_Vleft_new)
    H_Vleft.append(H_Vleft_new)

    H_right.append(Hright)
    I_Hright.append(I_Hright_new)
    H_Hright.append(H_Hright_new)

    # Binocular stage
    P_Vbin = max(0, mon_to_bin_gain * (Vleft + Hright) - I_wt_bin * I_Hbin[time_step])
    P_Hbin = max(0, mon_to_bin_gain * (Hright + Vleft) - I_wt_bin * I_Vbin[time_step])

    Vbin, I_Vbin_new, H_Vbin_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Vbin,
        V_bin[time_step], I_Vbin[time_step], H_Vbin[time_step],
        E_dot, I_dot, H_dot, dt
    )

    Hbin, I_Hbin_new, H_Hbin_new = RKThreeD(
        tau_E, tau_I, tau_H, P_Hbin,
        H_bin[time_step], I_Hbin[time_step], H_Hbin[time_step],
        E_dot, I_dot, H_dot, dt
    )

    V_bin.append(Vbin)
    H_bin.append(Hbin)
    I_Vbin.append(I_Vbin_new)
    I_Hbin.append(I_Hbin_new)
    H_Vbin.append(H_Vbin_new)
    H_Hbin.append(H_Hbin_new)

    time_step += 1

# --------------------
# Plotting: 3 Separate Subplots
# --------------------
plt.figure(figsize=(14, 5))

# 1. Left Monocular (Vertical)
# plt.subplot(3, 1, 1)
# plt.plot(time, V_left[1:], 'r', linewidth=2)
# plt.title('Left Monocular (Vertical)', fontsize=12)
# plt.ylabel('Firing Rate')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xlim(0, 10000)
# plt.ylim(0, 70)

# 2. Right Monocular (Horizontal)
# plt.subplot(3, 1, 2)
# plt.plot(time, H_right[1:], 'b', linewidth=2)
# plt.title('Right Monocular (Horizontal)', fontsize=12)
# plt.ylabel('Firing Rate')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xlim(0, 10000)
# plt.ylim(0, 70)

# Binocular Vertical and Horizontal
# plt.subplot(3, 1, 3)
plt.plot(time, V_bin[1:], 'r', linewidth=2, label='Binocular Vertical')
plt.plot(time, H_bin[1:], 'b', linewidth=2, label='Binocular Horizontal')
plt.title('Binocular Stage', fontsize=20)
plt.xlabel('Time (ms)')
plt.ylabel('Firing Rate')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, 10000)
plt.ylim(0, 70)

plt.tight_layout()
# plt.suptitle('Flicker-and-Swap (F&S) Rivalry: Monocular and Binocular Neurons', fontsize=14, y=1.03)
plt.show()
