# Returns the upper triangular part of a matrix (2-D tensor)

# torch.triu(input, diagonal=0, *, out=None) → Tensor
# The argument 'diagonal' controls which diagonal to consider.

import torch

source_tensor = torch.ones((10, 10))
# print(source_tensor)
tensor = (torch.triu(source_tensor) == 1).transpose(0, 1)
print(tensor)
print(tensor.float().masked_fill(tensor == 0, float('-inf')).masked_fill(tensor == 1, float(0.0)))
