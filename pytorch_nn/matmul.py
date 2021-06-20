# torch.matmul() with multi-dimensional tensor
# Requirement: The Last-dimension of the first tensor should match the Last-1 dimension of the second tensor

import torch

x = torch.arange(24).view(2, 2, 2, 3)
y = torch.arange(12).view(2, 3, 2)
print(x)
print(y)

m = torch.matmul(x, y)
print(m)
