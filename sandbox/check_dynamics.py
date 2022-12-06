"""
Quick script to verify hamiltonian and newtonian formulations of system dynamics produce the same transitions
"""


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

if __name__ == "__main__":
    Ix = Iy = 2.0
    dth1_0 = 0.5
    dth2_0 = -0.5
    p1_0 = dth1_0*Ix
    p2_0 = dth2_0*Iy
    dt = 0.01
    control = np.array([1, -1])

    n_x0 = np.array([1,1,dth1_0,dth1_0])
    h_x0 = np.array([1,1,p1_0,p2_0])

    ham_dyns = discretize_dynamics_rk4(hamiltonian_dynamics, dt)
    newt_dyns = discretize_dynamics_rk4(newtonian_dynamics, dt)

    for tt in range(5):
        h_x0 = ham_dyns(h_x0, control, dt)
        n_x0 = newt_dyns(h_x0, control, dt)
    
    print("Final Ham. State", h_x0)
    print("Final Newt State:", n_x0)
