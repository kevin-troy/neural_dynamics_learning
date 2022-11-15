import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
traj_th = np.loadtxt("C:/Users/kevin/Desktop/GitHub/neural_dynamics_learning/sandbox/th_traj.csv", delimiter=",",dtype=np.float64)
traj_th = traj_th[:,3]

l = 1
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-l*1.2, l*1.2)
ax.set_ylim(-l*1.2, l*1.2)
line, = ax.plot([0, l*np.cos(traj_th[0]+np.pi/2)], [0, l*np.sin(traj_th[0]+np.pi/2)])
comet, = ax.plot([1,2,3],[4,5,6], alpha=0.5)
ax.grid()
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Damped Pendulum Swingup")
fade_idx = 5
x = l*np.cos(traj_th+np.pi/2)
y = l*np.sin(traj_th+np.pi/2)
def animate(i):
    line.set_data([0,x[i]], [0,y[i]])
    if i <=fade_idx:
        comet.set_data(x[:i+1], y[:i+1])
    else:
        comet.set_data(x[i-fade_idx:i+1], y[i-fade_idx:i+1])

nframes = len(traj_th)
print(nframes)
interval = 100
anim = animation.FuncAnimation(fig, animate, frames=nframes, interval=interval, repeat=True)
plt.show()