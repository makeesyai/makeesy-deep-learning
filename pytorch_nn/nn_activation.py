import torch
from torch import nn
import matplotlib.pyplot as plt

x = torch.arange(start=-10, end=10, step=1).float()
# m = nn.Sigmoid()
# m = nn.Tanh()
# m = nn.ReLU()
# m = nn.LeakyReLU()
m = nn.GELU()
# m = nn.SiLU()
y = m(x)
print(y)
exit()
plt.plot(x.numpy(), y.squeeze().numpy())
plt.show()
