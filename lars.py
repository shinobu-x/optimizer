import torch
from torch.optim.optimizer import Optimizer, required

# Large Batch Training of Convolutional Networks
# https://arxiv.org/abs/1708.03888
class LARS(Optimizer):
  def __init__(self, params, lr = required, momentum = 0.1, dampening = 0.2,
      weight_decay = 1e-4, nesterov = False, eta = 1e-3, epsilon = 1e-3,
      max_epoch = 90):
    self.lr = lr
    self.eta = eta
    self.epsilon =  epsilon
    self.max_epoch = max_epoch
    defaults = dict(lr = lr, momentum = momentum, dampening = dampening,
        weight_decay = weight_decay, nesterov = nesterov)
    super(LARS, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(LARS, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('nesterov', False)

  @torch.no_grad()
  def step(self, epoch = None, closure = None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()
    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']
      dampening = group['dampening']
      nesterov = group['nesterov']
      lr = group['lr']
      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad
        weight_norm = torch.norm(p.data)
        gradient_norm = torch.norm(d_p.data)
        # Compute the global learning rate with polynomial decay
        global_lr = lr * (1 - float(epoch) / self.max_epoch) ** 2
        # Compute the local learning rate
        local_lr = min(self.eta * weight_norm / \
            (gradient_norm + weight_norm * weight_decay + self.epsilon), 1.0) \
            if weight_norm * gradient_norm > 0 else 1.0
        if weight_decay != 0.0:
          d_p = d_p.add(weight_decay, p)
        if momentum != 0.0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            buffer = param_state['momentum_buffer'] = torch.clone(d_p).detach()
          else:
            buffer = param_state['momentum_buffer']
          buffer.mul_(momentum).add_(1 - dampening, d_p)
          if nesterov:
            d_p = d_p.add(buffer, momentum)
          else:
            d_p = buffer
        p.add_(d_p, -global_lr * local_lr)
    return loss
