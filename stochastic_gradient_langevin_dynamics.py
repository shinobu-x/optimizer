import numpy as np
from torch import nn
from torch.optim.optimizer import Optimizer, required

class SGLD(Optimizer):
    def __init__(self, params, lr = required, sigma = 0.0, enable_noise = True):
        weight_decay = 1 / (sigma ** 2) if sigma != 0.0 else 0.0
        defaults = dict(lr = lr, weight_decay = weight_decay,
                enable_noise = enable_noise)
        super(SGLD, self).__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0.0:
                    d_p.add_(weight_decay, p.data)
                if group['enable_noise']:
                    noise = p.data.new(p.data.size()).normal_(mean = 0,
                            std = 1) / np.sqrt(group['lr'])
                    p.data.add_(-group['lr'], 0.5 * d_p + noise)
                else:
                    p.data.add_(-group['lr'], 0.5 * d_p)
        return loss
