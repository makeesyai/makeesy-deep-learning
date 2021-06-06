# Fills elements of the tensor with value where mask is True.
# Tensor.masked_fill_(mask, value) vs torch.masked_fill(tensor, mask, value)
import torch

tensor = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
])

# Full mask
mask = torch.tensor([
    [1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1]
])

print(tensor)
x = torch.masked_fill(tensor, mask, value=9999)
print(x)

# Row broadcasting
mask = torch.tensor([
    [1, 1, 0, 1, 1]
])

print(tensor)
y = torch.masked_fill(tensor, mask, value=9999)
print(y)

# Column broadcasting
mask = torch.tensor([
    [1],
    [0],
    [0],
])

print(tensor)
z = torch.masked_fill(tensor, mask, value=9999)
print(z)
