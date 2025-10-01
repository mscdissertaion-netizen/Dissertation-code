# -*- coding: utf-8 -*-
"""
@author: Chetan Mathias
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Constants ===
Ena = 0.5      # Sodium reversal potential
Ek = -0.95     # Potassium reversal potential
g = 26         # Potassium conductance
C = 1          # Membrane capacitance
I1 = I2 = I3 = I4 = 0.7  # Input to monocular neurons
I5 = I6 = 0.7          # external input to binocular neurons

# === Synaptic parameters ===
g_syn = 45.0    # Intra-pair inhibition strength
g_cross = 15.0  # Cross-pair inhibition strength
g_fb = 10.0     # Feedback inhibition from binocular neurons
g_bin = 45.0    # Mutual inhibition between binocular neurons
tau_syn = 2     # Synaptic time constant

# === Gating functions ===
def Rin(V): return 1.24 + 3.7 * V + 3.2 * (V ** 2)
def Tin(V): return 8 * ((V + 0.725) ** 2)
def m(V): return 17.8 + 47.6 * V + 33.8 * (V ** 2)
def heaviside(x): return 1.0 if x > 0 else 0.0

# === ODE system for 6 neurons with full dynamics ===
def rivalry_model_6neurons_full(t, y):
    V1, R1, T1, H1, f1, S1, \
    V2, R2, T2, H2, f2, S2, \
    V3, R3, T3, H3, f3, S3, \
    V4, R4, T4, H4, f4, S4, \
    V5, R5, T5, H5, f5, S5, \
    V6, R6, T6, H6, f6, S6 = y

    # === Monocular Neurons ===
    dV1_dt = (-m(V1)*(V1 - Ena) - g*R1*(V1 - Ek) - 0.1*T1*(V1 - 1.2) - 2.5*H1*(V1 - Ek)
              - g_syn*S2*(V1 - Ek) - g_cross*S3*(V1 - Ek) - g_fb*S6*(V1 - Ek) + I1) / C
    dR1_dt = (-R1 + Rin(V1)) / 1.5
    dT1_dt = (-T1 + Tin(V1)) / 50
    dH1_dt = (-H1 + 3*T1) / 900
    df1_dt = (-f1 + heaviside(V1 + 0.1)) / tau_syn
    dS1_dt = (-S1 + f1) / tau_syn

    dV2_dt = (-m(V2)*(V2 - Ena) - g*R2*(V2 - Ek) - 0.1*T2*(V2 - 1.2) - 2.5*H2*(V2 - Ek)
              - g_syn*S1*(V2 - Ek) - g_cross*S4*(V2 - Ek) - g_fb*S6*(V2 - Ek) + I2) / C
    dR2_dt = (-R2 + Rin(V2)) / 1.5
    dT2_dt = (-T2 + Tin(V2)) / 50
    dH2_dt = (-H2 + 3*T2) / 900
    df2_dt = (-f2 + heaviside(V2 + 0.1)) / tau_syn
    dS2_dt = (-S2 + f2) / tau_syn

    dV3_dt = (-m(V3)*(V3 - Ena) - g*R3*(V3 - Ek) - 0.1*T3*(V3 - 1.2) - 2.5*H3*(V3 - Ek)
              - g_syn*S4*(V3 - Ek) - g_cross*S1*(V3 - Ek) - g_fb*S5*(V3 - Ek) + I3) / C
    dR3_dt = (-R3 + Rin(V3)) / 1.5
    dT3_dt = (-T3 + Tin(V3)) / 50
    dH3_dt = (-H3 + 3*T3) / 900
    df3_dt = (-f3 + heaviside(V3 + 0.1)) / tau_syn
    dS3_dt = (-S3 + f3) / tau_syn

    dV4_dt = (-m(V4)*(V4 - Ena) - g*R4*(V4 - Ek) - 0.1*T4*(V4 - 1.2) - 2.5*H4*(V4 - Ek)
              - g_syn*S3*(V4 - Ek) - g_cross*S2*(V4 - Ek) - g_fb*S5*(V4 - Ek) + I4) / C
    dR4_dt = (-R4 + Rin(V4)) / 1.5
    dT4_dt = (-T4 + Tin(V4)) / 50
    dH4_dt = (-H4 + 3*T4) / 900
    df4_dt = (-f4 + heaviside(V4 + 0.1)) / tau_syn
    dS4_dt = (-S4 + f4) / tau_syn

    # === Binocular Neuron N5 ===
    dV5_dt = (-m(V5)*(V5 - Ena) - g*R5*(V5 - Ek) - 0.1*T5*(V5 - 1.2) - 2.5*H5*(V5 - Ek)
              + 1.2*(S1 + S2) - g_bin*S6*(V5 - Ek) - 0.05*V5 + I5) / C
    dR5_dt = (-R5 + Rin(V5)) / 1.5
    dT5_dt = (-T5 + Tin(V5)) / 50
    dH5_dt = (-H5 + 3*T5) / 900
    df5_dt = (-f5 + heaviside(V5 + 0.1)) / tau_syn
    dS5_dt = (-S5 + f5) / tau_syn

    # === Binocular Neuron N6 ===
    dV6_dt = (-m(V6)*(V6 - Ena) - g*R6*(V6 - Ek) - 0.1*T6*(V6 - 1.2) - 2.5*H6*(V6 - Ek)
              + 1.2*(S3 + S4) - g_bin*S5*(V6 - Ek) - 0.05*V6 + I6) / C
    dR6_dt = (-R6 + Rin(V6)) / 1.5
    dT6_dt = (-T6 + Tin(V6)) / 50
    dH6_dt = (-H6 + 3*T6) / 900
    df6_dt = (-f6 + heaviside(V6 + 0.1)) / tau_syn
    dS6_dt = (-S6 + f6) / tau_syn

    return [dV1_dt, dR1_dt, dT1_dt, dH1_dt, df1_dt, dS1_dt,
            dV2_dt, dR2_dt, dT2_dt, dH2_dt, df2_dt, dS2_dt,
            dV3_dt, dR3_dt, dT3_dt, dH3_dt, df3_dt, dS3_dt,
            dV4_dt, dR4_dt, dT4_dt, dH4_dt, df4_dt, dS4_dt,
            dV5_dt, dR5_dt, dT5_dt, dH5_dt, df5_dt, dS5_dt,
            dV6_dt, dR6_dt, dT6_dt, dH6_dt, df6_dt, dS6_dt]

# === Initial Conditions ===
y0 = [-0.75, 1.2, 0.03, 0.15, 0.0, 0.0,   # N1
      -0.7, 1.1, 0.02, 0.14, 0.0, 0.0,    # N2
      -0.72, 1.15, 0.025, 0.145, 0.0, 0.0,# N3
      -0.73, 1.18, 0.028, 0.148, 0.0, 0.0,# N4
      -0.6, 1.0, 0.02, 0.1, 0.0, 0.0,     # N5
      -0.6, 1.0, 0.02, 0.1, 0.0, 0.0]     # N6

# === Time Setup ===
t_start = 0
t_end = 6000
dt = 0.1
t_eval = np.arange(t_start, t_end, dt)
tspan = [t_start, t_end]

# === Solve System ===
sol = solve_ivp(rivalry_model_6neurons_full, tspan, y0, t_eval=t_eval)

# === Extract Voltages ===
V1 = sol.y[0] * 100
V2 = sol.y[6] * 100
V3 = sol.y[12] * 100
V4 = sol.y[18] * 100
V5 = sol.y[24] * 100
V6 = sol.y[30] * 100

# === Plot Membrane Potentials ===
plt.figure(figsize=(12, 6))
# plt.plot(sol.t, V1, label="N1 (Left Mono)")
# plt.plot(sol.t, V2, label="N2 (Left Mono)")
# plt.plot(sol.t, V3, label="N3 (Right Mono)")
# plt.plot(sol.t, V4, label="N4 (Right Mono)")
plt.plot(sol.t, V5, label="N5 (Binocular L)")
plt.plot(sol.t, V6, label="N6 (Binocular R)")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
# plt.title("6-Neuron Rivalry with Full Binocular Dynamics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Synaptic Outputs ===
S1 = sol.y[5]
S2 = sol.y[11]
S3 = sol.y[17]
S4 = sol.y[23]
S5 = sol.y[29]
S6 = sol.y[35]

plt.figure(figsize=(12, 5))
# plt.plot(sol.t, S1, label="S1 (N1)")
# plt.plot(sol.t, S2, label="S2 (N2)")
# plt.plot(sol.t, S3, label="S3 (N3)")
# plt.plot(sol.t, S4, label="S4 (N4)")
plt.plot(sol.t, S5, label="S5 (N5 Binocular L)", linestyle='--')
plt.plot(sol.t, S6, label="S6 (N6 Binocular R)", linestyle='--')
plt.xlabel("Time (ms)")
plt.ylabel("Synaptic Output")
# plt.title("Synaptic Activity of All Neurons")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

