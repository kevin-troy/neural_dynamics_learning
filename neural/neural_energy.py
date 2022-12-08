import torch
import torch.nn as nn
from torch.autograd import grad as grad
import pytorch_lightning as pl
import torch.utils.data as data
from torch.distributions import Uniform, Normal
from torchdyn.models import ODEProblem

"""
[1] Massaroli et. al., Optimal Energy Shaping via Neural Approximators, https://arxiv.org/pdf/2101.05537.pdf
[2] TorchDyn tutorials, https://github.com/DiffEqML/torchdyn/tree/master/tutorials

"""

# System physical parameters
Ix = 2.0
Iy = 2.0
device = 'cuda'

class MirrorSystem(nn.Module):
    def __init__(self, V, K):
        super().__init__()
        self.V, self.K = V, K
        self.Ix, self.Iy = Ix, Iy
    
    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            q = x[:,0:2]
            p = x[:,2:4]
            q.requires_grad_(True)

            u = self._energy_shaping(q) + self._damping_injection(x)
            dxdt = self._dynamics(q,p,u)
        return dxdt
        
    def _dynamics(self, q, p, u):
        # Continuous time Hamiltonian dynamics
        # Returns shape [N_train x 4]
        dq1 = p[:,0]/Ix
        dq2 = p[:,1]/Ix
        dp1 = u[:,0]
        dp2 = u[:,1]
        output = torch.stack([dq1,dq2,dp1,dp2], 1)
        assert(output.shape[0] == p.shape[0])
        assert(output.shape[1] == 4)

        return output

    def _energy_shaping(self, q):
        # Energy shaping portion of control output
        # [1], eq. 16, first term
        # Returns [N_train x 2] tensor
        dVdx = grad(self.V(q).sum(), q, create_graph=True)[0]
        
        assert(dVdx.shape[0] == q.shape[0])
        assert(dVdx.shape[1] == 2)

        return -dVdx

    def _damping_injection(self, x):
        # Damping injection portion of control output
        # [1], eq. 16, second term
        # Returns [N_train x 2] tensor
        assert(x.shape[1] == 4)        
        output = self.K(x)*x[:, 2:]/torch.Tensor([self.Ix, self.Iy]).to(x)
        assert(output.shape[0] == x.shape[0])
        assert(output.shape[1] == 2)

        return output

    def _autonomous_energy(self, x):
        """
        Hamiltonian of uncontrolled system
        Returns [N_train x 1] tensor
        """
        energy = 1/2*Ix*x[:,0]**2 + 1/2*Iy*x[:,1]**2
        
        assert(energy.shape[0] == x.shape[0])
        assert(energy.shape[1] == 1)

        return energy

    def _energy(self, x):
        """
        Hamiltonian of controlled system
        Returns [N_train x 1] tensor
        """
        control_energy = self.V(x[:,:2])

        assert(control_energy.shape[0] == x.shape[0])
        assert(control_energy.shape[1] == 1)

        return self._autonomous_energy(x) + control_energy 


class AugmentedMirror(nn.Module):
    def __init__(self, f, integral_loss):
        super().__init__()
        self.f = f
        self.integral_loss = integral_loss
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        x = x[:,:4]
        
        dxdt = self.f(t,x)   
        dLdt = self.integral_loss(t,x)
        output = torch.cat([dxdt, dLdt], 1)
        assert(output.shape[1] == 5)

        return output


class EnergyShapingLearner(pl.LightningModule):
    def __init__(self, model: nn.Module, prior_dist, target_dist, t_span, lr=5e-3, batch_size=2048, gamma=0.999, sensitivity='autograd'):
        super().__init__()
        self.model = model
        self.prior = prior_dist
        self.target = target_dist
        self.t_span = t_span
        self.batch_size = batch_size
        self.lr = lr
        self.weight = torch.tensor([1., 1., 1., 1.]).reshape(1,4)
        self.gamma = gamma

    def forward(self, x):
        #print("odeIntXShape=", x.shape)
        return self.model.odeint(x, self.t_span)
    
    def training_step(self, batch, batch_idx):
        # Sample ICs
        x0 = self.prior.sample((self.batch_size,))
        # [Batch_size x N_states = 4]

        # Integrate
        x0 = torch.cat([x0, torch.zeros(self.batch_size,1).to(x0)], 1)
        # x0.shape -> [Batch_size x N_states + 1 = 5]
        _,xTl = self(x0)
        xT, L = xTl[-1:, :, :4], xTl[-1,:,-1:]

        # Compute loss
        terminal_loss = weighted_log_likelihood_loss(xT, self.target, self.weight.to(xT))
        integral_loss = torch.mean(L)

        loss = terminal_loss + 0.0001*integral_loss
        self.log("val_loss", loss)
        return {'loss':loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        temp =  data.DataLoader(
            data.TensorDataset(torch.Tensor(1,1), torch.Tensor(1,1), torch.Tensor(1,1), torch.Tensor(1,1)),
            batch_size = 1)
        return temp


def prior_distribution(q1_min, q1_max, q2_min, q2_max, p1_min, p1_max, p2_min, p2_max, device='cuda'):
    lower = torch.Tensor([q1_min, q2_min, p1_min, p2_min]).to(device)
    upper = torch.Tensor([q1_max, q2_max, p1_max, p2_max]).to(device)
    return Uniform(lower, upper)


def posterior_distribution(mu, sigma, device='cuda'):
    mu, sigma = torch.Tensor(mu).reshape(1,4).to(device), torch.Tensor(sigma).reshape(1,4).to(device)
    return Normal(mu, torch.sqrt(sigma))


def weighted_log_likelihood_loss(x, target, weight):
    # weighted negative log likelihood loss
    log_prob = target.log_prob(x)
    weighted_log_p = weight * log_prob
    return -torch.mean(weighted_log_p.sum(1))


class ControlEffort(nn.Module):
    def __init__(self,f):
        super().__init__()
        self.f = f
    def forward(self,t, x):
        with torch.set_grad_enabled(True):
            q = x[:,:2].requires_grad_(True)
            u = self.f._energy_shaping(q) + self.f._damping_injection(x)
        output = torch.sum(torch.abs(u),1, keepdim=True)
        return output


def train_energy_model(V, K, t_end, N_timesteps, prior, target, batch_size=2048, lr=5e-3, gamma=0.999, epochs=100, use_early_stopping=False):
    # Initialize final linear layers as zeros
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    for p in V[-1].parameters(): torch.nn.init.zeros_(p)
    for p in K[-2].parameters(): torch.nn.init.zeros_(p)

    # Define dynamics
    f = MirrorSystem(V,K)
    aug_f = AugmentedMirror(f, ControlEffort(f))

    # Define time horizon
    t_span = torch.linspace(0,t_end,N_timesteps)
    problem = ODEProblem(aug_f, sensitivity='autograd', solver='rk4')
    learn = EnergyShapingLearner(problem, prior, target, t_span, lr=lr, batch_size=batch_size, gamma=gamma)
    
    if use_early_stopping:
        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
    else:   
        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu")

    trainer.fit(learn)
    model = aug_f.cpu()
    return model


def rollout_plots(model, t_end, N_timesteps, prior, desired):
    import matplotlib.pyplot as plt
    from torchdiffeq import odeint
    import numpy as np

    t_span = torch.linspace(0, t_end, N_timesteps)
    global device
    device = 'cpu'
    n_ic = 100
    x0 = prior.sample(torch.Size([n_ic])).cpu()
    x0 = torch.cat([x0, torch.zeros(n_ic,1)], 1)

    traj = odeint(model, x0, t_span, method='midpoint').detach()
    traj = traj[..., :-1]

    fig, ax = plt.subplots(4,1)
    labels = ["q_1", "q_2", "p_1", "p_2"]
    for i in range(n_ic):
        for state_id in range(4):
            ax[state_id].plot(t_span, traj[:,i,0], 'k', alpha=.1)
            ax[state_id].plot(t_span, torch.ones_like(traj[:,i,0])*desired[state_id], 'k', alpha=.1)
            ax[state_id].set_xlabel("Time (s)")
            ax[state_id].set_ylabel(labels[state_id])

    plt.show()
