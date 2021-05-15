# Fills elements of self tensor with value where mask is True.
# Tensor.masked_fill_(mask, value)

import torch

tensor = torch.tensor([[1, 2, 3, 4],
                       [4, 5, 6, 7]],
                      dtype=torch.float)
# Full mask
mask = torch.tensor([[True, False, True, False],
                     [False, True, True, False]])
tensor_masked = tensor.masked_fill(mask, float('0.0'))
print(tensor_masked)

# Row broadcasting
mask = torch.tensor([True, False, True, False])
tensor_masked = tensor.masked_fill(mask, float('0.0'))
print(tensor_masked)

# Column broadcasting
mask = torch.tensor([[True], [False]])
tensor_masked = tensor.masked_fill(mask, float('0.0'))
print(tensor_masked)
