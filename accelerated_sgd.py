from copy import deepcopy
from torch.optim.optimizer import Optimizer, required

# Accelerating Stochastic Gradient Descent For Least Squares Regression
# https://arxiv.org/abs/1704.08227
class AcceleratedSGD(Optimizer):
    def __init__(self, params, lr = required, kappa = 10 ** 5, xi = 10.0,
            constant = 0.7, weight_decay = 0.0):
        defaults = dict(lr = lr, kappa = kappa, xi = xi, constant = constant,
                weight_decay = weight_decay)
        super(AcceleratedSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AcceleratedSGD, self).__setstate__(state)

    def step(self, closure = None):
        loss = None
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = (group['lr'] * group['kappa']) / (group['constant'])
            alpha = 1.0 - ((group['constant'] ** 2 * group['xi']) /
                    group['kappa'])
            beta = 1.0 - alpha
            zeta = group['constant'] / (group['constant'] + beta)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0.0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = deepcopy(p.data)
                buffer = param_state['momenbum_buffer']
                buffer.mul_((1.0 / beta) - 1.0).add_(-lr, d_p).add_(p.data)
                buffer.mul_(beta)
                p.data.add_(-group['lr'], d_p).mul_(zeta)
                p.data.add_(1.0 - zeta, buffer)
        return loss
