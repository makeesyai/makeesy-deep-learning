# Operations on Tensors
# 1. Arithmetic operations (explain broadcasting: row/column, and single-element tensors)
# torch.ops(t1, t2, t_out:optional), t1.ops(t2): element-wise/scalar/broadcasting
# 2. In-place operations
# t1.ops_(t2): element-wise/scalar/broadcasting

import torch

x = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float32)
y = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float32)
z = torch.Tensor()
torch.add(x, y, out=z)
print(z)
z = x.add(y)
print(z)
# z = x.add_(y)
# print(z)
# print(x)
# print(x == z)

# scalar
z = x.add(2)
print(z)

# row broadcasting
tensor = torch.tensor([1, 2, 3])
z = x.add(tensor)
print(z)

# column broadcasting
tensor = torch.tensor([[1],
                       [2]])
z = x.add(tensor)
print(z)

# 3. Joining tensors
x = torch.tensor([[1, 2, 3], [1, 2, 3]])
y = torch.tensor([[1, 2, 3], [1, 2, 3]])
z = torch.cat([x, y], dim=1)
print(z)
