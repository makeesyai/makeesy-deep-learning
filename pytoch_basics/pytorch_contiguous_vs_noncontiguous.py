# It’s a flag indicating, if the memory is contiguously stored or not.
# Continuous arrays are necessary for some vectorized instructions to work e.g. view().
# use is_contiguous() method to check if a tensor is contiguous for the cases it is required to have.

# Create a tensor of shape [4, 3]
import torch

tensor = torch.arange(12).view(4, 3)
print(tensor)
print(tensor.stride())

tensor = tensor.t()
print(tensor)
print(tensor.stride())

# tensor = tensor.view(-1)  # Throw a Runtime error
tensor = tensor.contiguous().view(-1)
print(tensor)
print(tensor.stride())
