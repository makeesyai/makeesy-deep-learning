import torch

x = torch.arange(6).view(2, 3)
y = torch.arange(6).view(3, 2)
print(x)
print(y)

m = torch.matmul(x, y)
print(m)
