import numpy as np
import torch.nn as nn
import torch 
from torchdiffeq import odeint
import time

def sim_neural(model:nn.Module, state_initial, desired, t_span):
    time_st = time.time()
    n_ic, n_states = state_initial.shape
    assert(n_states == 4)
    x0 = torch.cat([state_initial, torch.zeros((n_ic,1))],1)
    traj = odeint(model, x0, t_span, method='midpoint').detach()
    traj = traj[...,:-1]

    u_shaped = torch.cat([model.f._energy_shaping(q.requires_grad_()) for q in traj[:,:,:2]],1).detach()
    u_diss = torch.cat([model.f._damping_injection(x) for x in traj], 1).detach()
    u = u_shaped + u_diss

    time_end = time.time()
    t_elapsed_s = time_end-time_st
    return traj, u, t_elapsed_s