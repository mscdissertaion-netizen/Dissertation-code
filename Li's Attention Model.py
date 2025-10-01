# -*- coding: utf-8 -*-
"""
@author: Chetan Mathias
"""

import numpy as np
import matplotlib.pyplot as plt

class NModel:
    """
    A neural network model for Li's attention model.
    This model implements a multi-layer neural network to simulate various visual conditions dissucssed in the dissertation.
    """
    
    def __init__(self, cond):
        """
        Initialise the model with a specific condition.
        
        Args:
            cond (int): Condition identifier (1-7) specifying the experimental condition
        """
        # Set model parameters based on condition
        self.p = self.set_parameters(cond)
        # Initialise time series arrays for all variables
        self.init_time_series()
        # Set up the stimulus pattern based on condition
        self.set_stim()

    def set_parameters(self, cond):
        """
        Define and configure model parameters based on the experimental condition.
        
        Args:
            cond (int): Condition identifier (1-7)
            
        Returns:
            dict: Dictionary containing all model parameters
        """
        p = {}
        p['cond'] = cond
        p['condnames'] = ['Attended rivalry', 'Unattended rivalry', 'Attended plaid',
                          'Unattended plaid', 'Blank before swap', 'Swap with no blank', 'Flicker-and-Swap']
        p['input'] = [0.5, 0.5]
        # Time step and total simulation time
        p['dt'] = 0.5  
        p['T'] = 15000  
        p['nt'] = int(p['T'] / p['dt']) + 1 s
        p['tlist'] = np.arange(0, p['T'] + p['dt'], p['dt'])  
        
        # Spatial and orientation dimensions
        p['nx'] = 1  
        p['ntheta'] = 2  

        # Model architecture parameters
        p['nLayers'] = 6  # total number of layers in the model
        p['rectSmoothFlag'] = 1  # use smooth rectification if 1
        p['n_m'] = 1  # exponent for monocular layers
        p['n'] = 2  # exponent for binocular layers
        p['sigma'] = 0.5  # semi-saturation constant
        p['m'] = 2  # response scaling factor
        p['wh'] = 2  # weight for habituative feedback
        p['wo'] = 0.65  # weight for opponency inhibition
        p['wa'] = 0.6  # weight for attention
        
        # Time constants for different processes (in milliseconds)
        p['tau_s'] = 5  # synaptic time constant
        p['tau_o'] = 20  # opponency time constant
        p['tau_a'] = 150  # attention time constant
        p['tau_h'] = 2000  # habituation time constant

        # Attention layer parameters
        p['sigma_a'] = 0.2  # semi-saturation for attention
        p['aKernel'] = np.array([[1, -1], [-1, 1]])  # attention kernel for cross-orientation inhibition
        p['alpha_t'] = 3  # time constant for stimulus onset
        p['alphaAmp'] = 0.5  # amplitude of onset transient
        p['tan_t'] = 30  # time constant for offset transient

        # Condition-specific parameter adjustments
        if cond == 2 or cond == 4:
            p['wa'] = 0  # no attention for unattended conditions
        elif cond == 5:
            p['ISP'] = 333  # inter-stimulus period
            p['blank'] = 150  # blank duration before swap
        elif cond == 6:
            p['ISP'] = 333
            p['blank'] = 0  # no blank before swap
        elif cond == 7:
            p['fHz'] = 18  # flicker frequency
            p['sHz'] = 3  # swap frequency

        return p

    def init_time_series(self):
        """
        Initialise arrays to store the time evolution of all model variables.
        Each layer has multiple state variables that evolve over time.
        """
        p = self.p
        
        # Dictionary to store time series for each variable type per layer
        p['d'] = {}  # drive input
        p['s'] = {}  # suppressive pool
        p['r'] = {}  # firing rate
        p['f'] = {}  # normalised response
        p['h'] = {}  # habituative state
        p['o'] = {}  # opponency inhibition

        # Initialise arrays for each layer
        for lay in range(1, p['nLayers'] + 1):
            # Create 2D arrays: orientations × time points
            p['d'][lay] = np.zeros((p['ntheta'], p['nt']))
            p['s'][lay] = np.zeros((p['ntheta'], p['nt']))
            p['r'][lay] = np.zeros((p['ntheta'], p['nt']))
            p['f'][lay] = np.zeros((p['ntheta'], p['nt']))
            p['h'][lay] = np.zeros((p['ntheta'], p['nt']))
            # Opponency layers only exist for layers 1 and 2
            if lay in [1, 2]:
                p['o'][lay] = np.zeros((p['ntheta'], p['nt']))

        # Small random initial conditions for key layers
        p['r'][1][:, 0] = np.random.rand(p['ntheta']) * 0.2
        p['r'][4][:, 0] = np.random.rand(p['ntheta']) * 0.2

    def set_stim(self):
        """
        Create stimulus time series based on the experimental condition.
        Defines how stimuli are presented to the left and right eyes over time.
        """
        p = self.p
        nt = p['nt']  
        dt = p['dt']  
        theta = p['ntheta']  
        tlist = p['tlist']  

        def makealpha(dt, T, tau, bound=1e-3):
            """
            Create an alpha function (rise and fall) for stimulus onsets.
            
            Args:
                dt: time step
                T: total duration
                tau: time constant
                bound: cutoff threshold
                
            Returns:
                numpy array: alpha function waveform
            """
            t = np.arange(0, T + dt, dt)
            alpha = (t / tau) * np.exp(1 - t / tau)  # alpha function formula
            alpha[(t > tau) & (alpha < bound)] = 0  # truncate small values
            return alpha[alpha > 0]  # return only non-zero portion

        def makeoffset(dt, duration):
            """
            Create an offset transient function using tanh.
            
            Args:
                dt: time step
                duration: duration of offset
                
            Returns:
                numpy array: offset waveform
            """
            x = np.linspace(np.pi - 1, -np.pi + 0.4, int(duration / dt + 2))
            y = 0.5 * np.tanh(x) + 0.5  # tanh-based offset function
            y_normalised = (y - y.min()) / (y.max() - y.min())  # normalise to [0,1]
            return y_normalised[1:-1]  # remove endpoints

        # Condition 1-2: Binocular rivalry (different orientations to each eye)
        if p['cond'] in [1, 2]:
            mod = np.ones(nt)  # baseline modulation
            onset = makealpha(dt, p['alpha_t'] * 100, p['alpha_t'])
            mod[:len(onset)] += onset * p['alphaAmp']  # adds onset transient
            
            # Orientation preferences: A to left eye, B to right eye
            oriL = np.array([[1], [0]])  # left eye prefers orientation A
            oriR = np.array([[0], [1]])  # right eye prefers orientation B
            
            # Create stimulus matrices with onset modulation
            p['stimL'] = oriL @ np.ones((1, nt)) * p['input'][0] * np.tile(mod, (2, 1))
            p['stimR'] = oriR @ np.ones((1, nt)) * p['input'][1] * np.tile(mod, (2, 1))

        # Condition 3-4: Plaid perception (same orientations to both eyes)
        elif p['cond'] in [3, 4]:
            mod = np.ones(nt)
            onset = makealpha(dt, p['alpha_t'] * 100, p['alpha_t'])
            mod[:len(onset)] += onset * p['alphaAmp']
            
            # Both orientations to left eye, nothing to right eye (plaid configuration)
            oriL = np.array([[1], [1]])  # left eye gets both orientations
            oriR = np.array([[0], [0]])  # right eye gets nothing
            
            p['stimL'] = oriL @ np.ones((1, nt)) * p['input'][0] * np.tile(mod, (2, 1))
            p['stimR'] = oriR @ np.ones((1, nt)) * p['input'][1] * np.tile(mod, (2, 1))

        # Condition 5-6: Stimulus swapping with/without blank period
        elif p['cond'] in [5, 6]:
            ts_L = np.zeros((theta, nt))  # left eye stimulus time series
            ts_R = np.zeros((theta, nt))  # right eye stimulus time series
            
            # Create state sequence (1 or 2) that alternates every ISP
            ts_state = (np.floor(tlist / p['ISP']) % 2 + 1).astype(int)
            onsetIdx = np.abs(np.concatenate(([1], np.diff(ts_state))))  # detect state changes
            ts_alpha = np.convolve(onsetIdx, makealpha(dt, p['alpha_t'] * 100, p['alpha_t']))[:nt]
            
            mod = np.ones((theta, nt))
            changeIdx = np.where(np.abs(np.diff(ts_state)) > 0)[0]  # indices where state changes
            
            # Create blank period before swaps if specified
            nblank = round(p['blank'] / dt)
            for i in range(1, nblank + 1):
                mod[:, changeIdx - i] = 0  # set stimulus to zero during blank
            
            # Create offset transients
            offsetIdx = (np.diff(mod[0, :], prepend=1) == -1).astype(float)
            ts_offset = np.convolve(offsetIdx, makeoffset(dt, p['tan_t']))[:nt]
            
            # Combine baseline, onset, and offset modulations
            mod += np.outer(np.ones(theta), ts_alpha * p['alphaAmp'] + ts_offset)
            
            # Define orientation patterns for each state
            s1 = np.array([[1], [0]])  # state 1: A to left, B to right
            s2 = np.array([[0], [1]])  # state 2: B to left, A to right
            
            # Assign orientations based on current state
            for t in range(nt):
                if ts_state[t] == 1:
                    ts_L[:, t] = s1.flatten()
                    ts_R[:, t] = s2.flatten()
                else:
                    ts_L[:, t] = s2.flatten()
                    ts_R[:, t] = s1.flatten()
                    
            p['stimL'] = ts_L * mod * p['input'][0]
            p['stimR'] = ts_R * mod * p['input'][1]

        # Condition 7: Flicker-and-swap paradigm
        elif p['cond'] == 7:
            ts_L = np.zeros((theta, nt))
            ts_R = np.zeros((theta, nt))
            
            s1 = np.array([[1], [0]])  # orientation pattern for state 1
            s2 = np.array([[0], [1]])  # orientation pattern for state 2
            
            # Create flicker pattern (on/off cycles)
            nframe_on = int(round(1000 / p['fHz'] / dt / 2))  # frames per half-cycle
            flickerIdx = np.tile(np.concatenate([np.ones(nframe_on), np.zeros(nframe_on)]),
                                 int(np.ceil(nt / (2 * nframe_on))))[:nt]
            
            # Detect flicker onsets and offsets
            onsetIdx = (np.diff(np.insert(flickerIdx, 0, 0)) == 1).astype(float)
            offsetIdx = (np.diff(np.insert(flickerIdx, 0, 0)) == -1).astype(float)
            
            # Create modulation signals
            alpha = makealpha(dt, p['alpha_t'] * 100, p['alpha_t'])[:nframe_on]
            ts_alpha = np.convolve(onsetIdx, alpha)[:nt]  # onset transients
            ts_offset = np.convolve(offsetIdx, makeoffset(dt, p['tan_t']))[:nt]  # offset transients
            
            # Combine flicker with transients
            flicker = np.tile(flickerIdx + ts_alpha * p['alphaAmp'] + ts_offset, (theta, 1))
            
            # Create slower swap rhythm
            swapCycle = p['fHz'] / p['sHz']  # how many flicker cycles per swap
            ts_state = (np.floor(np.arange(nt) / (swapCycle * nframe_on * 2)) % 2 + 1).astype(int)
            
            # Assign orientations based on swap state
            for t in range(nt):
                if ts_state[t] == 1:
                    ts_L[:, t] = s1.flatten()
                    ts_R[:, t] = s2.flatten()
                else:
                    ts_L[:, t] = s2.flatten()
                    ts_R[:, t] = s1.flatten()
                    
            p['stimL'] = ts_L * flicker * p['input'][0]
            p['stimR'] = ts_R * flicker * p['input'][1]

    def run_model(self):
        """
        Run the neural network simulation through time.
        Updates all layer activities based on differential equations.
        """
        p = self.p

        def half_exp(x, n=1):
            """Simple half-wave rectification with exponent."""
            return np.maximum(0, x) ** n

        def half_exp_smooth(x):
            """
            Smooth half-wave rectification using sigmoid function.
            Provides continuous approximation of rectification.
            """
            thresh = 0.05  # threshold for smooth transition
            slope = 30  # slope of sigmoid
            y = np.zeros_like(x)
            idx_pos = x > 0
            # Smooth transition using sigmoid function
            y[idx_pos] = x[idx_pos] * (1 / (1 + np.exp(-slope * (x[idx_pos] - thresh))))
            return y

        # Choose rectification function based on parameter
        h = half_exp_smooth if p['rectSmoothFlag'] else half_exp

        # Main simulation loop through time
        for idx in range(1, p['nt']):
            # Progress indicator every 5000 ms
            if abs(p['tlist'][idx] % 5000) < p['dt'] / 2:
                print(f"{p['tlist'][idx]:.0f} msec")

            # MONOCULAR LAYERS (1-2) 
            for lay in [1, 2]:
                # Get appropriate stimulus (left eye for layer 1, right for layer 2)
                stim = p['stimL'][:, idx] if lay == 1 else p['stimR'][:, idx]
                # Inhibition from opponency layers
                inhibition = p['o'][lay][:, idx - 1] if lay in p['o'] else np.zeros(p['ntheta'])
                # Feedback from attention layer
                attention = p['r'][6][:, idx - 1] if 6 in p['r'] else np.zeros(p['ntheta'])
                # Drive = stimulus - inhibition, modulated by attention
                excitatory = h(stim ** p['n_m'] - inhibition * p['wo']) * h(1 + attention * p['wa'])
                p['d'][lay][:, idx] = excitatory

            for lay in [1, 2]:
                # Pool inputs from both monocular layers for normalisation
                pool = np.hstack((p['d'][1][:, idx], p['d'][2][:, idx]))
                p['s'][lay][:, idx] = np.sum(pool)  # suppressive pool
                # Normalisation denominator
                denom = p['s'][lay][:, idx] + p['sigma'] ** p['n_m'] + p['h'][lay][:, idx - 1] ** p['n_m']
                # Normalised response
                p['f'][lay][:, idx] = p['m'] * p['d'][lay][:, idx] / denom
                # Update firing rate using differential equation
                p['r'][lay][:, idx] = p['r'][lay][:, idx - 1] + (p['dt'] / p['tau_s']) * (
                    -p['r'][lay][:, idx - 1] + p['f'][lay][:, idx])
                # Update habituation state
                p['h'][lay][:, idx] = p['h'][lay][:, idx - 1] + (p['dt'] / p['tau_h']) * (
                    -p['h'][lay][:, idx - 1] + p['r'][lay][:, idx - 1] * p['wh'])

            # BINOCULAR AND OPPONENCY LAYERS (3-5) 
            for lay in [3, 4, 5]:
                if lay == 3:  # Binocular summation layer
                    inp = p['r'][1][:, idx - 1] + p['r'][2][:, idx - 1]  # sum both eyes
                    p['d'][lay][:, idx] = inp ** p['n']
                elif lay == 4:  # Opponency layer for left > right
                    diff = p['r'][1][:, idx - 1] - p['r'][2][:, idx - 1]
                    p['d'][lay][:, idx] = h(diff) ** p['n']  # rectified difference
                elif lay == 5:  # Opponency layer for right > left
                    diff = p['r'][2][:, idx - 1] - p['r'][1][:, idx - 1]
                    p['d'][lay][:, idx] = h(diff) ** p['n']

            for lay in [3, 4, 5]:
                if lay == 3:  # Binocular summation layer
                    p['s'][lay][:, idx] = p['d'][lay][:, idx]
                    denom = p['s'][lay][:, idx] + p['sigma'] ** p['n'] + p['h'][lay][:, idx - 1] ** p['n']
                    p['f'][lay][:, idx] = p['d'][lay][:, idx] / denom
                    p['r'][lay][:, idx] = p['r'][lay][:, idx - 1] + (p['dt'] / p['tau_s']) * (
                        -p['r'][lay][:, idx - 1] + p['f'][lay][:, idx])
                    p['h'][lay][:, idx] = p['h'][lay][:, idx - 1] + (p['dt'] / p['tau_h']) * (
                        -p['h'][lay][:, idx - 1] + p['r'][lay][:, idx] * p['wh'])
                else:  # Opponency layers
                    pool = p['d'][lay][:, idx]
                    p['s'][lay][:, idx] = np.sum(pool)
                    denom = p['s'][lay][:, idx] + p['sigma'] ** p['n']
                    p['f'][lay][:, idx] = p['d'][lay][:, idx] / denom
                    p['r'][lay][:, idx] = p['r'][lay][:, idx - 1] + (p['dt'] / p['tau_o']) * (
                        -p['r'][lay][:, idx - 1] + p['f'][lay][:, idx])

                    # Send inhibition to monocular layers
                    if lay == 4:  # Inhibit right eye monocular layer
                        p['o'][2][:, idx] = np.sum(p['r'][lay][:, idx])
                    elif lay == 5:  # Inhibit left eye monocular layer
                        p['o'][1][:, idx] = np.sum(p['r'][lay][:, idx])

            # ATTENTION LAYER (6) 
            inp = p['r'][3][:, idx]  # input from binocular layer
            # Cross-orientation interactions using kernel
            aDrive = np.abs(p['aKernel'] @ inp)  # absolute drive
            aSign = np.sign(p['aKernel'] @ inp)  # sign for excitation/inhibition
            excitatory = aSign * (aDrive ** p['n'])
            p['d'][6][:, idx] = excitatory
            
            suppressive = np.sum(aDrive ** p['n']) + p['sigma_a'] ** p['n']  # normalisation pool
            p['s'][6][:, idx] = suppressive
            p['f'][6][:, idx] = p['d'][6][:, idx] / suppressive  # normalised attention
            # Update attention state
            p['r'][6][:, idx] = p['r'][6][:, idx - 1] + (p['dt'] / p['tau_a']) * (
                -p['r'][6][:, idx - 1] + p['f'][6][:, idx])

        print("Simulation complete.")

    def plot_time_series(self):
        """
        Plot the time evolution of key model variables.
        Shows responses from different layers to visualise the dynamics.
        """
        p = self.p
        # Color matrix for different orientations
        colmat = [[0, 0.5, 0], [0, 0, 1]]  # green for A, blue for B
        
        t = p['tlist'] / 1000  # convert to seconds
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)
        
        # Plot 1: Binocular summation layer (combined perception)
        axs[0].plot(t, p['r'][3][0, :], color=colmat[0])  # orientation A
        axs[0].plot(t, p['r'][3][1, :], color=colmat[1])  # orientation B
        axs[0].legend(['Orientation A', 'Orientation B'])
        axs[0].set_ylim(0, 1)
        axs[0].set_ylabel('Response')
        axs[0].set_title('Binocular-summation Layer')
        
        # Plot2: Left eye monocular layer
        axs[1].plot(t, p['r'][1][0, :], color=colmat[0])
        axs[1].plot(t, p['r'][1][1, :], color=colmat[1])
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel('Response')
        axs[1].set_title('Left-eye monocular layer')
        
        # Plot 3: Right eye monocular layer
        axs[2].plot(t, p['r'][2][0, :], color=colmat[0])
        axs[2].plot(t, p['r'][2][1, :], color=colmat[1])
        axs[2].set_ylim(0, 1)
        axs[2].set_xlabel('Time (sec)')
        axs[2].set_ylabel('Response')
        axs[2].set_title('Right-eye monocular layer')
        
        # temp1 = np.maximum(p['r'][6][0, :] + 1, 0)
        # temp2 = np.maximum(p['r'][6][1, :] + 1, 0)
        
        ## plot 4 : Attenion gain
        # axs[0].plot(t, temp1, color=colmat[0])
        # axs[0].plot(t, temp2, color=colmat[1])
        # axs[0].set_ylabel('Attentional gain')
        # axs[0].set_title('Attentional gain')
        
        
        ## plot 5 : Mutual inhibition
        # axs[1].plot(t, p['o'][1][0, :], 'k-')
        # axs[1].plot(t, p['o'][2][0, :], 'k--')
        # axs[1].set_ylim(0, 0.8)
        # axs[1].set_xlabel('Time (sec)')
        # axs[1].set_ylabel('Mutual inhibition')
        # axs[1].set_title('Inhibition from opponency layers')
        # axs[1].legend(['supp from R', 'supp from L'])
        plt.show()


# Run the model and plot the results
model = NModel(cond=7)  # Create model with condition 7 (Flicker-and-Swap)
model.run_model()       # Run the simulation

model.plot_time_series()  # Plot the results
