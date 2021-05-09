# Operations on Tensors
# 1. Arithmetic operations (explain broadcasting: row/column, and single-element tensors)
# torch.ops(t1, t2, t_out:optional), t1.ops(t2): element-wise/scalar/broadcasting
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
y = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

# torch.ops(t1, t2)
z = torch.add(x, y)
print(z)
z = torch.Tensor()

# torch.ops(t1, t2, out=t3)
torch.add(x, y, out=z)
print(z)

# t1.ops(t2)
z = x.add(y)
print(z)

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

# Singleton out
m = torch.max(x)
print(m.item())

# 2. In-place operations
# t1.ops_(t2): element-wise/scalar/broadcasting
z = x.add_(y)
print(z)
print(x)
print(x == z)

# 3. Joining tensors
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
y = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])
z = torch.cat([x, y])
print(z)
z = torch.cat([x, y], dim=1)
print(z)
