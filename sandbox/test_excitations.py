
import matplotlib.pyplot as plt
from excitation_policies import *

null_exciter = null_excitation()
sine_exciter = sine_excitation(1, 2, 0)
step_exciter = step_excitation(2)
impulse_exciter = impulse_excitation(2.5)

t = np.linspace(0,5,1000)
x = None
controls = np.zeros((5,len(t)))
for idx,tstep in enumerate(t):
    controls[0,idx] = null_exciter(tstep,x)
    controls[1,idx] = sine_exciter(tstep,x)
    controls[2,idx] = step_exciter(tstep,x)
    controls[3,idx] = impulse_exciter(tstep,x)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(5):
    ax.plot(t, controls[i,:])
plt.show()