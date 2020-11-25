import torch
from torch.optim.optimizer import Optimizer, required

# Stochastic Mirror Descent on Overparameterized Nonlinear Models: Convergence,
# Implicit Regularization, and Generalization
# https://arxiv.org/abs/1906.03830
class StochasticMirrorDescent(Optimizer):
    def __init__(self, params, lr = required, momentum = 0.0,
            weight_decay = 0.0, dampening = 0.0, epsilon = 0.1,
            enable_compress = False):
        defaults = dict(lr = lr, momentum = momentum,
                weight_decay = weight_decay, dampening = dampening)
        super(StochasticMirrorDescent).__init__(params, defaults)
        self.epsilon = epsilon
        self.enable_compress = enable_compress

    def __setstate__(self, state):
        super(StochasticMirrorDescent).__setstate__(state)

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0.0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buffer = param_state['momentum_buffer'] = \
                                torch.zeros_like(p.data)
                        buffer.mul_(momentum).add_(d_p)
                    else:
                        buffer = param_state['momentum_buffer']
                        buffer.mul(momentum).add_(1 - dampening, d_p)
                    d_p = buffer
                if self.enable_compress:
                    update = (1.0 + self.epsilon) * \
                            (torch.abs(p.data) ** self.epsilon) * \
                            torch.sign(p.data) - group['lr'] * d_p
                    p.data = (torch.abs(update / (1.0 + self.epsilon)) ** \
                            (1 / self.epsilon)) * torch.sign(update)
                else:
                    update = (group['q']) * \
                            (torch.abs(p.data) ** (group['q'] - 1)) * \
                            torch.sign(p.data) - group['lr'] * d_p
                    p.data = (torch.abs(
                        update / group['q'])) ** (1 / (group['q'] - 1)) * \
                                torch.sign(update)
        return loss
