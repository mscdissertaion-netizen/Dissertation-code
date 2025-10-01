# -*- coding: utf-8 -*-
"""
@author: Chetan Mathias
"""

import numpy as np
from numpy.random import default_rng
import math
import matplotlib.pyplot as plt

#  Model Parameters 
# These parameters control the dynamics of the mutual inhibition system
betav = 10  # Sensitivity parameter for v's effect on u
betau = 10  # Sensitivity parameter for u's effect on v
hu = 0.5    # Threshold parameter for u activation
hv = 0.5    # Threshold parameter for v activation
tu = 25     # Time constant for u decay
tv = 25     # Time constant for v decay

# Start with u dominant (u population has initial advantage)
u_dominant = True

def v_func(betav, v, hv, tu, tv, u_dominant):
    """
    Rate function for u activation. 
    This represents how population v inhibits population u.
    Uses a sigmoidal response function.
    """
    if u_dominant:
        return 1 / (1 + np.exp(betav * ((v / tv) - hv)))
    else:
        return 1 / (1 + np.exp(betav * ((v / tv) - hv)))

def u_func(betau, u, hu, tu, tv, u_dominant):
    """
    Rate function for v activation.
    This represents how population u inhibits population v.
    Uses a sigmoidal response function.
    """
    if u_dominant:
        return 1 / (1 + np.exp(betau * ((u / tu) - hu)))
    else:
        return 1 / (1 + np.exp(betau * ((u / tu) - hu)))

#  Simulation Parameters 
tmax = 100000      # Maximum simulation time
nt = 200000        # Maximum number of time steps
rng_seed = 2       # Random number generator seed for reproducibility

# Initial conditions
u0 = 18  # Initial value for u population
v0 = 0   # Initial value for v population

# Initialise time and state variables arrays
t = np.zeros(nt)
variables = np.zeros((2, nt))  # 2 rows for u and v, nt columns for time steps
variables[:, 0:1] = np.array([[u0], [v0]])  # Set initial conditions

rng = default_rng(rng_seed)  # Initialise random number generator

# Stoichiometric matrix defines how each reaction affects the populations
# Columns represent reactions: [u_activation, u_decay, v_activation, v_decay]
# Rows represent populations: [u, v]
stoichiometry = np.array([
    [1, -1, 0,  0],  # u increases by 1 when activated, decreases by 1 when decaying
    [0,  0, 1, -1]   # v increases by 1 when activated, decreases by 1 when decaying
])

# Stimulus Parameters 
stimulus_times = [3000, 40000, 20000]  # Times when stimuli are applied
stimulus_duration = 200                 # How long each stimulus lasts
stimulus_strength = 50                  # Strength of the stimulus boost
suppressed_threshold = 30               # Threshold below which population is considered suppressed

# Track dominance and stimulus usage
current_dominance = u_dominant  # Start with initial dominance state
stimuli_used = [False for _ in stimulus_times]  # Track which stimuli have been applied

# Gillespie Simulation Loop 
# The Gillespie algorithm is a stochastic simulation method for chemical reactions
j = 0  # Time step counter
while t[j] < tmax and j < nt - 1:
    u = variables[0, j]  # Current u population
    v = variables[1, j]  # Current v population
    
    # Check if any stimulus is currently active
    stimulus_active = False
    for i, st in enumerate(stimulus_times):
        if st < t[j] < (st + stimulus_duration):
            stimulus_active = True
            # Switch dominance only once per stimulus
            if not stimuli_used[i]:
                current_dominance = not current_dominance
                stimuli_used[i] = True
            break

    # Calculate reaction rates based on current state
    rates = np.array([
        v_func(betav, v, hv, tu, tv, current_dominance),   # activation of u
        u / tu,                                            # decay of u (linear decay)
        u_func(betau, u, hu, tu, tv, current_dominance),   # activation of v
        v / tv                                             # decay of v (linear decay)
    ])
    
    # Apply stimulus boost only to the suppressed population
    if stimulus_active:
        if current_dominance:  
            # u is dominant, boost v if it's suppressed
            if v < suppressed_threshold:
                rates[2] += stimulus_strength
        else:  
            # v is dominant, boost u if it's suppressed
            if u < suppressed_threshold:
                rates[0] += stimulus_strength

    # Gillespie algorithm steps:
    rtot = rates.sum()  # Total reaction rate
    
    # Calculate time until next reaction (exponentially distributed)
    t[j + 1] = t[j] - math.log(1 - rng.uniform()) / rtot
    
    # Determine which reaction occurs next
    rtot_rand = rtot * rng.uniform()  # Random number to select reaction
    r = rates.cumsum()  # Cumulative sum of rates
    reaction = np.searchsorted(r, rtot_rand)  # Find which reaction occurs
    
    # Update population counts based on the reaction
    variables[:, j + 1] = variables[:, j] + stoichiometry[:, reaction]
    j += 1  # Move to next time step

# Trim arrays to actual simulation length
t = t[0:j]
u = variables[0, 0:j]
v = variables[1, 0:j]

# Plot Results 
plt.figure(figsize=(14, 10))  

# Plot u population over time
plt.subplot(211)
plt.step(t, u, where='post', label='u(t)')
# Mark stimulus times with vertical lines
for st in stimulus_times:
    plt.axvline(x=st, color='r', linestyle='--', alpha=0.5)
plt.ylabel('u(t)', fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  

# Plot v population over time
plt.subplot(212)
plt.step(t, v, where='post', label='v(t)')
# Mark stimulus times with vertical lines
for st in stimulus_times:
    plt.axvline(x=st, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Time', fontsize=16)  
plt.ylabel('v(t)', fontsize=16)
plt.xticks(fontsize=14)  
plt.yticks(fontsize=14)  

plt.tight_layout()

plt.show()
