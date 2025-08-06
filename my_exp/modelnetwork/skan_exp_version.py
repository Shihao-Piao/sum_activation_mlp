import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch

def lrelu(x, k):
    return torch.clamp(k*x, min=0)

def fixed_sin(x,k):
    w = 1 #fixed w
    return k * torch.sin(w*x)

def random_sin(x,k,w):
    return k * torch.sin(w*x)

def ranged_sin(x,k,w):
    return k * torch.sin(w*x)


class SKANLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, base_function=lrelu, device='cpu'):
        super(SKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.base_function = base_function
        self.device = device
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features+1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
    
    def forward(self, x):
        x = x.view(-1, 1, self.in_features)
        # 添加偏置单元
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)

        y = self.base_function(x, self.weight)
        
        y = torch.sum(y, dim=2)
        return y
    
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
class SKANNetwork(nn.Module):
    def __init__(self, layer_sizes, base_function=lrelu, bias=True, device='cpu'):
        super(SKANNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(SKANLinear(layer_sizes[i], layer_sizes[i+1], bias=bias, 
                                             base_function=base_function, device=device))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SKANLinear_random(nn.Module):
    def __init__(self, in_features, out_features, bias=True, base_function=random_sin, device='cpu'):
        super(SKANLinear_random, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.base_function = base_function
        self.device = device
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features + 1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()
        self.w = nn.Parameter(
            torch.empty(out_features, in_features + (1 if bias else 0)),
            requires_grad=False
        )
        #nn.init.uniform_(self.w, 0.5, 3.0)
        nn.init.normal_(self.w, mean=0.0, std=1.0)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        x = x.view(-1, 1, self.in_features)
        # 添加偏置单元
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)

        y = self.base_function(x, self.weight,self.w)

        y = torch.sum(y, dim=2)
        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class SKANNetwork_random(nn.Module):
    def __init__(self, layer_sizes, base_function=lrelu, bias=True, device='cpu'):
        super(SKANNetwork_random, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(SKANLinear_random(layer_sizes[i], layer_sizes[i + 1], bias=bias,
                                          base_function=base_function, device=device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SKANLinear_ranged(nn.Module):
    def __init__(self, in_features, out_features, bias=True, base_function=random_sin, device='cpu'):
        super(SKANLinear_ranged, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.base_function = base_function
        self.device = device
        if bias:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features + 1).to(device))
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.reset_parameters()
        # One frequency per output node
        w = torch.arange(1, out_features + 1, dtype=torch.float32)  # shape: [out_features]
        w = w.view(-1, 1).repeat(1, in_features + (1 if bias else 0))  # shape: [out_features, in_features]

        w *= 2 * np.pi /out_features # modify

        self.w = nn.Parameter(w, requires_grad=False)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        x = x.view(-1, 1, self.in_features)
        # 添加偏置单元
        if self.use_bias:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=2)

        y = self.base_function(x, self.weight,self.w)

        y = torch.sum(y, dim=2)
        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class SKANNetwork_ranged(nn.Module):
    def __init__(self, layer_sizes, base_function=lrelu, bias=True, device='cpu'):
        super(SKANNetwork_ranged, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(SKANLinear_ranged(layer_sizes[i], layer_sizes[i + 1], bias=bias,
                                          base_function=base_function, device=device))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x