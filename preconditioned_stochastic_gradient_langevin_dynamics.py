import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

class PSGLD(Optimizer):
    def __init__(self, params, lr = required, sigma = 0.0, alpha = 0.99,
            eps = 1e-8, centered = False, enable_noise = True):
        weight_decay = 1 / (sigma ** 2) if sigma != 0.0 else 0.0
        defaults = dict(lr = lr, weight_decay = weight_decay, alpha = alpha,
                eps = eps, centered = centered, enable_noise = enable_noise)
        super(PSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self):
        loss = None
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                   continue
                d_p = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, d_p)
                    avg = square_avg.cuml(-1, grad_avg, grad_avg).sqrt().add_(
                            group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                if group['enable_noise']:
                    noise = p.data.new(p.data.size()).normal_(mean = 0.0,
                            std = 1.0) / np.sqrt(group['lr'])
                    p.data.add_(-group['lr'], 0.5 * d_p.div_(avg) +
                            noise / torch.sqrt(avg))
                else:
                    p.data.addcdiv_(-group['lr'], 0.5 * d_p, avg)
        return loss
