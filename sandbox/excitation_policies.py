"""
A series of functions to generate control inputs. Used to sample the dynamics of a system for training data.
"""

import numpy as np

def lqr_controller(Q,R):
    # Defines an LQR feedback controller for the cartpole
    raise NotImplementedError


def pid_controller(kp, ki, kd):
    # Defines a PID controller for the cartpole
    raise NotImplementedError


"""
Sinusoidal Excitation: Returns a parameterized sinusoidal control
"""
class sine_excitation:
    def __init__(self, amplitude, frequency, phase):
        self.params = (amplitude, frequency, phase)
    def __call__(self, t, x):
        A,f,p = self.params
        return A*np.sin(2*np.pi*f*t+p)


"""
Null Excitation: Returns zero control for all timesteps
"""
class null_excitation:
    def __call__(self, t, x):
        return 0.0


"""
Impulse Excitation: Returns a control of specified amplitude at t=0, and zero for all other t.
"""
class impulse_excitation:
    def __init__(self, amplitude=1.0):
        self.triggered = False
        self.amplitude = amplitude
    def __call__(self, t, x):
        if not self.triggered:
            self.triggered = True
            return self.amplitude
        else:
            return 0.0
    def reset(self):
        self.triggered = False
        

"""
Step Excitation: Returns a control of zero at t=0, and parameter "amplitude" for all other t.
"""
class step_excitation:
    def __init__(self, amplitude=1.0):
        self.triggered = False
        self.amplitude = amplitude
    def __call__(self, t, x):
        if not self.triggered:
            self.triggered = True
            return 0.0
        else:
            return self.amplitude
    def reset(self):
        self.triggered = False




