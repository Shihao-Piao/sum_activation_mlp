'''
import torch

# Create input with gradient tracking
x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)

# Optimizer-like step size
lr = 0.1

# ReLU activation
relu = torch.nn.ReLU()

# Forward
y = relu(x) ** 2
loss = y.sum()

# Backward
loss.backward()
print(x.grad)

# Update (gradient descent step)
x = x - lr * x.grad

# Optional: zero the grad manually if doing more steps
x.requires_grad_()  # re-enable gradient tracking if needed again

print("Updated x:", x)

import torch.nn as nn
class ActivatedWeightedSum(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU()):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = activation

    def forward(self, x):
        # x: (batch_size, input_dim)
        print(x.shape , self.weight.shape)
        x = x.unsqueeze(1)                      # -> (batch_size, 1, input_dim)
        print("x:",x.shape,x)
        w = self.weight.unsqueeze(0)            # -> (1, output_dim, input_dim)
        print("w:",w.shape,w)
        out = self.activation(x * w)            # -> (batch_size, output_dim, input_dim)
        print("first out:",out.shape,out)
        out = out.sum(dim=2) + self.bias        # sum over input_dim, then add bias
        return out                              # -> (batch_size, output_dim)

mlp = ActivatedWeightedSum(2,3)

x = torch.tensor([[-2.0,  0.5]])
print(mlp(x))
'''
import torch
import torch as th
import numpy as np

outdim = 3
inputdim = 2
gridsize = 4

grid_norm_factor =  np.sqrt(gridsize)
re = th.randn(outdim,inputdim)/ (np.sqrt(inputdim) * grid_norm_factor )

print(re)

k = th.reshape( th.arange(1,gridsize+1,device='cpu'),(1,1,1,gridsize))

print(k)

para_k = th.nn.Parameter(re)

print(para_k)

t = torch.tensor([1.,2.])
t = torch.nn.Parameter(t)

print(t)

r = torch.nn.Parameter(
    torch.empty(outdim, inputdim),
    requires_grad=False
)

r = torch.nn.init.uniform_(r,-2,2)

print(r)

w_base = torch.arange(1, 4).view(-1, 1)
# w_base =
# [[1.0],
#  [2.0],
#  [3.0]]

# With repeat(1, 4):
w = w_base.repeat(3, 2)
print(w)