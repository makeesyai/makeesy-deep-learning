# Why do we need activation functions?
# 1. To add non-linearity to the network
# 2. To add stability in the training as in some cases the variance is to high and using activations we can
# restrict them to a range {0, 1} or {-1, 1} etc.
# 3. Control the information flow in the network

# Commonly used: Sigmoid, Tanh, ReLU, LeakyReLU/PReLU, ELU

import torch
from torch import nn
import matplotlib.pyplot as plt

x = torch.arange(start=-10, end=10, step=0.1).float()
m = nn.Sigmoid()
# m = nn.Tanh()
# m = nn.ReLU()
# m = nn.LeakyReLU()
# m = nn.PReLU()
# m = nn.ELU()

y = m(x)
plt.plot(x.numpy(), y.detach().squeeze().numpy())
plt.show()
