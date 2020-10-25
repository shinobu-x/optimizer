import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer, SGD

# Optimizing Neural Networks with Kronecker-factored Approximate Curvature
# https://arxiv.org/abs/1503.05671
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias

class SplitBias(nn.Module):
    def __init__(self, model):
        super(SplitBias, self).__init__()
        self.model = model
        self.add_bias = AddBias(model.bias.data)
        self.model.bias = None

    def forward(self, x):
        x = self.model(x)
        x = self.add_bias(x)
        return x

class KFAC(Optimizer):
    def __init__(self, model, lr = 0.01, momentum = 0.9, kl_clip = 1e-3,
            cnn = False, stat_decay = 1e-3, damping = 1e-2, t1 = 1, t2 = 10,
            weight_decay = 0.0):
        default = dict()
        def split_bias(module):
            for name, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[name] = SplitBias(child)
                else:
                    split_bias(child)
        split_bias(model)
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.model = model
        self.default_modules = {'Linear', 'Conv2d', 'AddBias'}
        self.modules = []
        self.grad_outputs = {}
        self.prepare_model()
        self.cnn = cnn
        self.stat_decay = stat_decay
        self.steps = 0
        self.matrix_a = {}
        self.matrix_g = {}
        self.Q_a = {}
        self.Q_g = {}
        self.d_a = {}
        self.d_g = {}
        self.lr = lr
        self.momentum = momentum
        self.kl_clip = kl_clip
        self.t1 = t1
        self.t2 = t2
        self.optim = SGD(model.parameters(), lr = self.lr,
                momentum = self.momentum)

    def extract(self, x, kernel_size, stride, padding):
        if padding[0] + padding[1] > 0:
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]
                )).data
            x = x.unfold(2, kernel_size[0], stride[0])
            x = x.unfold(3, kernel_size[1], stride[1])
            x = x.transpose_(1, 2)
            x = x.transpose_(2, 3).contiguous()
            x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) *
                    x.size(4) * x.size(5))
        return x

    def compute_covariance_a(self, a, classname, layer_info, cnn):
        batch_size = a.size(0)
        if classname == 'Conv2d':
            if cnn:
                a = self.extract(a, *layer_info)
                a = a.view(a.size(0), -1, a.size(-1))
                a = a.mean(1)
            else:
                a = self.extract(a, *layer_info)
                a = a.view(-1, a.size(-1))
                a = a.div_(a.size(1)).div_(a.size(2))
        elif classname == 'AddBias':
            a = torch.ones(a.size(0), 1).to(self.device)
        return a.t() @ (a / batch_size)

    def compute_covariance_g(self, g, classname, layer_info, cnn):
        batch_size = g.size(0)
        if classname == 'Conv2d':
            if cnn:
                g = g.view(g.size(0), g.size(1), -1)
                g = g.sum(-1)
            else:
                g = g.transpose(1, 2).transpose(2, 3).contiguous()
                g = g.view(-1, g.size(-1))
                g = g.mul_(g.size(1)).mul_(g.size(2))
        elif classname == 'AddBias':
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        return (g * batch_size).t() @ ((g * batch_size) / g.size(0))

    def update_running_stat(self, x, matrix_x):
        matrix_x *= momentum / (1 - momentum)
        matrix_x *= x
        matrix_x *= (1 - momentum)

    def save_input(self, module, input):
        if torch.is_grad_enable() and self.steps % self.t1 == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                        module.padding)
            covariance = compute_covariance_a(input[0].data, classname,
                    layer_info, self.cnn)
            if self.steps == 0:
                self.matrix_a[module] = covariance.clone()
            update_running_stat(covariance, self.matrix_a[module],
                    self.stat_decay)

    def save_gradient_output(self, module, grad_input, grad_output):
        classname = module.__class__.__name__
        layer_info = None
        if classname == 'Conv2d':
            layer_info = (module.kernel_size, module.stride,
                    module.padding)
        covariance = compute_covariance_g(grad_output[0].data, classname,
                layer_info, self.cnn)
        if self.steps == 0:
            self.matrix_g[module] = covariance.clone()
        update_running_stat(covariance, self.matrix_g[module],
                self.stat_decay)

    def prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.default_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self.save_input)
                module.register_backward_hook(self.save_grad_output)

    def step(self):
        if self.weight_decay > 0:
            for param in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)
        updates = {}
        for _, module in enumerate(self.modules):
            classname = module.__class__.__name__
            param = next(module.parameters())
            if self.steps % self.t2 == 0:
                self.d_a[module], self.Q_a[module] = torch.symeig(
                        self.matrix_a[module], eigenvectors = True)
                self.d_g[module], self.Q_g[module] = torch.symeig(
                        self.matrix_g[module], eigenvectors = True)
                self.d_a[module].mul_((self.d_a[module] > 1e-6).float())
                self.d_g[module].mul_((self.d_g[module] > 1e-6).float())
            if classname == 'Conv2d':
                matrix_grad = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                matrix_grad = p.grad.data
            x = self.Q_g[module].t() @ matrix_grad @ self.Q_a[module]
            x = x / (self.d_g[module].unsqueeze(1) *
                    self.d_a[module].unsqueeze(1) *
                    (self.damping + self.weight_decay))
            x = self.Q_g[module] @ x @ self.Q_a[module].t()
            x = x.view(p.grad.data.size())
            updates[param] = x
        sum_xgradient = 0
        for param in self.model.parameters():
            x = updates[param]
            sum_xgradient += (x * param.grad.data * self.lr ** 2).sum()
        nu = min(1, math.sqrt(self.kl_clip / sum_xgradients))
        for param in self.model.parameters():
            x = updates[param]
            param.grad.data.copy_(x)
            param.grad.data_mul_(nu)
        self.optim.step()
        self.steps += 1
