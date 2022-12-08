import neural_energy
import torch.nn as nn
from numpy import pi as pi

hdim = 64
V = nn.Sequential(
    nn.Linear(2, hdim),
    nn.ReLU(),
    nn.Linear(hdim, hdim),
    nn.ReLU(),
    nn.Linear(hdim, hdim),
    nn.Tanh(),
    nn.Linear(hdim,2)
)
K = nn.Sequential(
    nn.Linear(4, hdim),
    nn.ReLU(),
    nn.Linear(hdim, hdim),
    nn.ReLU(),
    nn.Linear(hdim,2),
    nn.ReLU()
)

t_end, N_timesteps = 10, 100

desired = [0.,0.,0.,0.]
prior = neural_energy.prior_distribution(-pi/8, pi/8, -pi/8, pi/8, -pi/16, pi/16, -pi/16, pi/16)
target = neural_energy.posterior_distribution(desired, [0.001, 0.001, 0.001, 0.001])
model = neural_energy.train_energy_model(V, K, t_end, N_timesteps, prior, target, batch_size=1024, epochs=100)
neural_energy.rollout_plots(model, t_end, N_timesteps, prior, desired)
