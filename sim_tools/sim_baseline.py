import numpy as np
import dynamics_tools.dynamics as dynamics

def sim_mpc(t, state_initial, state_desired, Q, R, P, N):
    """
    Simulates the baseline MPC with linearized newtonian dynamics
    """
    dt = t[1]-t[0]
    A, B = dynamics.linear_dynamics(dt)

    # Init vars
    state_history = np.zeros((t.shape[0], 4))
    control_history = np.zeros((t.shape[0],2))

    # Set initial state
    state_history[0,:] = state_initial

    # Get full dynamics
    f = dynamics.discretize_dynamics_rk4(dynamics.newtonian_dynamics, dt) 

    # Sim time
    for tstep,tt in enumerate(t):
        if tstep == t.shape[0]-1:
            break
        control_history[tstep,:],_,_ = dynamics.mpc_step(state_history[tstep,:], state_desired, A, B, P, Q, R, N)
        state_history[tstep+1,:] = f(state_history[tstep,:], control_history[tstep,:], dt)

    return state_history, control_history