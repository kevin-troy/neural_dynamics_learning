import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("Hello world")

def proportional_controller(x):
    return 0.0*x[0]

def pend(t, y, b, c, control_policy):
    theta, omega = y
    dydt = [omega, control_policy(y) -b*omega + c*np.sin(theta)]
    print("Control=", control_policy(y))
    return dydt

x0 = np.array([0.1, 0.0])
t = np.linspace(0.0, 10, 10000)
dt = t[1]-t[0]
b = 5.0
c = 2500.0

sol = solve_ivp(pend, [t[0], t[-1]], x0, args = (b,c,proportional_controller), t_eval = t)

print(sol.y.shape)
print(sol.t)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
L = 1
ax.set_xlim(-L*1.2, L*1.2)
ax.set_ylim(-L*1.2, L*1.2)
line, = ax.plot([0, np.cos(x0[0]+np.pi/2)], [0, np.sin(x0[0]+np.pi/2)])

def animate(i):
    x = np.cos(sol.y[0,i]+np.pi/2)
    y = np.sin(sol.y[0,i]+np.pi/2)
    line.set_data([0,x], [0,y])

nframes = len(sol.y[0])
interval = dt/1000
anim = animation.FuncAnimation(fig, animate, frames=nframes, interval=interval, repeat=True)

plt.show()