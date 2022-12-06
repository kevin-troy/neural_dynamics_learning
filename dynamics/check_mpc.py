import numpy as np
import matplotlib.pyplot as plt
import dynamics

t = np.linspace(0,10, 100)
dt = t[1]-t[0]
Q = 1*np.eye(4)
R = 1*np.eye(2)
P = np.diag([100, 100, 50, 50])
N = 20
A,B = dynamics.linear_dynamics(dt)

# Init vars
state_history = np.zeros((t.shape[0], 4))
control_history = np.zeros((t.shape[0],2))

# Initial state
state_history[0,:] = np.array([np.pi/8, np.pi/8, 0, 0])
state_desired = np.array([0.0, 0.0, 0.0, 0.0])

# Full system
f = dynamics.discretize_dynamics_rk4(dynamics.newtonian_dynamics, dt) 
OPEN_LOOP = False

for tstep,tt in enumerate(t):
    if tstep == t.shape[0]-1:
        break
    if OPEN_LOOP:
        control_history[tstep,:] = np.array([0.0, 0.0])
        state_history[tstep+1,:] = f(state_history[tstep,:], control_history[tstep,:], dt)
    else:
        control_history[tstep,:],_,_ = dynamics.mpc_step(state_history[tstep,:], state_desired, A, B, P, Q, R, N)
        state_history[tstep+1,:] = f(state_history[tstep,:], control_history[tstep,:], dt)

print("plot time")

# Plot state trajectories
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(t, state_history[:,0], 'b')
ax.plot(t, state_history[:,1], 'g')
ax.plot(t, state_desired[0]*np.ones_like(t), 'b--')
ax.plot(t, state_desired[1]*np.ones_like(t), 'g--')

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(t, control_history[:,0])
ax.plot(t, control_history[:,1])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(t, state_history[:,0] - state_desired[0])
ax.plot(t, state_history[:,1] - state_desired[1])

plt.show()