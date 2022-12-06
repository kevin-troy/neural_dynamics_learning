import numpy as np

def discretize_dynamics_rk4(f:callable, dt:float):
    def integrator(state, control, dt):
        k1 = dt*f(state,control)
        k2 = dt*f(state+k1/2,control)
        k3 = dt*f(state+k2/2,control)
        k4 = dt*f(state+k3, control)
        return state+(k1+2*k2+2*k3+k4)/6
    return integrator

def newtonian_dynamics(state, control):
    Ix = Iy = 2.0
    th1, th2, th1d, th2d = state
    T1, T2 = control
    return np.array([th1d,
                     th2d,
                     T1/Ix,
                     T2/Iy])

def hamiltonian_dynamics(state, control):
    Ix = Iy = 2.0
    q1, q2, p1, p2 = state
    T1, T2 = control
    return np.array([p1/Ix,
                     p2/Iy,
                     T1,
                     T2])

def linear_dynamics(dt):
    # Euler integrated linear dynamics for use in MPC
    Ix = Iy = 2.0
    A = np.array([[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])
    B = np.array([[0, 0],[0,0], [dt/Ix,0], [0, dt/Iy]])
    return A, B

