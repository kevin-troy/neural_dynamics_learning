import numpy as np
import cvxpy as cvx

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

def mpc_step(s0, sg, A, B, P, Q, R, N):
    """
    Computes a single iteration of MPC
    :param s0: Initial State
    :param sg: Goal State
    :param A: Dynamics A matrix
    :param B: Dynamics B matrix
    :param P: Terminal state cost matrix
    :param Q: State deviation cost matrix
    :param R: Control cost matrix
    :param N: Lookahead horizon (steps)
    :return u*: Optimal control for next timestep
    :return status: Optimization problem status
    :return s_cvx: State solution to optimization
    :return u_cvx: Control solution to optimization
    """

    n = Q.shape[0]
    m = R.shape[0]

    s_cvx = cvx.Variable((N+1, n))
    u_cvx = cvx.Variable((N, m))

    constraints = [s_cvx[0] == s0]                      # Initial state constraint
    objective = cvx.quad_form(s_cvx[-1]-sg, P)          # Terminal state cost

    for k in range(N):
        objective += cvx.quad_form(s_cvx[k]-sg, Q)      # State deviation cost
        objective += cvx.quad_form(u_cvx[k], R)         # Control cost

        # Dynamics constraint
        constraints += [s_cvx[k+1, :] == A@s_cvx[k, :] + B@u_cvx[k, :]]

    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    problem.solve()

    if problem.status != 'optimal':
        raise RuntimeError("MPC Solver Failed. Status = ", problem.status)
    ustar = u_cvx.value[0]

    return ustar, s_cvx.value, u_cvx.value