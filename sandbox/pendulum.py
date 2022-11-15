import numpy as np
import torch
from scipy.integrate import odeint

def generate_pendulum(X0, t_eval, control_policy, noise=1e-1, mass=1, length=1, gravity=9.81):

    # Pendulum equations of motion
    # X = [theta, theta_dot]
    def odefunc(t,x,tau,params):
        m,l,g = params
        return np.array([x[1], tau-m*g*np.cos(x[0])]).T
    
    X = []
    for x0 in X0:
        sol = odeint(odefunc, [t_eval[0], t_eval[-1]], x0, method='LSODA', t_eval=t_eval)
        print(sol)

def lqr_controller(Q,R):
    # Defines an LQR feedback controller for the cartpole
    raise NotImplementedError

def pid_controller(kp, ki, kd):
    # Defines a PID controller for the cartpole
    raise NotImplementedError

def impulse(magnitude=1, at_timestep=0):
    # Excites the system with an impulse or series of impulses
    if "calls" not in impulse.__dict__: impulse.calls = 0

    if impulse.calls in at_timestep:
        out = magnitude
    else:
        out = 0.0

    impulse.calls+=1
    return out

def heaviside_step(magnitude=1, at_timestep=0):
    # Excites the system with a step
    raise NotImplementedError

def sine(amplitude=1, frequency=1, phase=1):
    raise NotImplementedError

def null_controller()
    # Returns zero control
    raise NotImplementedError