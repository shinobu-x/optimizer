import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class BayesByBackprop(Optimizer):
    def __init__(self, params, lr = 1e-2, beta = 0.99, rho = 3.0,
            epsilon = 1e-3):
        super(BayesByBackprop, self).__init__(params,
                defaults = dict(lr = lr, beta = beta, epsilon = epsilon))
        self.lr = lr
        self.beta = beta
        self.rho = rho
        self.epsilon = epsilon
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['rho_prior'] = torch.ones_like(p.data)
                state['mu_prior'] = p.data.clone()
                state['rho'] = torch.ones_like(p.data) * self.rho
                state['mu'] = p.data.clone()
                state['mu_g_sq'] = torch.ones_like(p.data)
                state['rho_g_sq'] = torch.ones_like(p.data)
        self.init_accumulators()

    def init_accumulators(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['epsilon'] = torch.ones_like(p.data)
                state['grad_mu'] = torch.zeros_like(p.data)
                state['grad_rho'] = torch.zeros_like(p.data)
                state['mc_iters'] = 0

    def set_weights(self, force_std = False):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if force_std:
                    p.data.copy_(state['mu'])
                else:
                    state['epsilon'] = torch.normal(torch.zeros_like(p.data),
                            torch.zeros_like(p.data))
                    std = F.softplus(state['rho'])
                    w = state['mu'] + state['epsilon'].mul(std)
                    p.data.copy_(w)

    def aggregate_grads(self, num_batches):
        for group in self.param_groups:
            if p.grad is None:
                continue
            state = self.state[p]
            assert state['epsilon'] is not None
            state['mc_iters'] += 1
            w = p.data
            mu = state['mu'].data
            mu_prior = state['mu_prior'].data
            rho = state['rho'].data
            rho_prior = state['rho_prior'].data
            std = F.softplus(rho)
            std_prior = F.softplus(rho_prior)
            state['grad_mu'] += p.grad.data * num_batches + (w - mu_prior) / \
                    (std_prior) ** 2
            state['grad_rho'] += ((-1e-1 / std) + std / (std_prior ** 2) + \
                    p.grad.data * num_batches * state['epsilon'].data) / \
                    (1 + torch.exp(-rho))
            state['epsilon'] = None

    def update_prior(self):
        for group in self.param_groups:
            for p in group['params']:
                state['mu_prior'].copy_(state['mu'])
                state['rho_prior'].copy_(state['rho'])

    def step(self, closure = None):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['mc_iters'] <= 0:
                    continue
                self.lr = group['lr']
                self.beta = group['beta']
                self.epsilon = group['epsilon']
                state['mu_g_sq'] = state['mu_g_sq'] * self.beta + \
                        (state['grad_mu'] ** 2) * (1 - self.beta)
                d_mu = -self.lr * state['grad_mu'] / state['mc_iters'] / \
                        (state['mu_g_sq'] + self.epsilon)
                state['mu'].data += d_mu
                state['rho_g_sq'] = state['rho_g_sq'] * self.beta + (
                        state['grad_rho'] ** 2) * (1 - self.beta)
                d_rho = -self.lr * state['grad_rho'] / state['mc_iters'] / \
                        (state['rho_g_sq'] + self.epsilon)
                state['rho'].data += d_rho
        self.init_accumulators()
