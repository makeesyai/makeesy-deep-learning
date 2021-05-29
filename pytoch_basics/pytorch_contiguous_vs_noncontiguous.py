# It’s a flag indicating, if the memory is contiguously stored or not.
# Continuous arrays are necessary for some operation to work e.g. tensor.view().

# use is_contiguous() method to check if a tensor is contiguous for the cases it is required to have.
# Use tensor.contiguous(): returns itself if input tensor is already contiguous, otherwise it
# returns a new contiguous tensor by copying data.

# General Operations that may returns non-contiguous: narrow() , view() , expand() or transpose() etc.
import torch

# Create a tensor of shape [4, 3]
tensor = torch.arange(12).view(4, 3)
print(tensor.is_contiguous())
print(tensor)
print(tensor.stride())

# Transpose the tensor to get a non-contiguous tensor
tensor = tensor.t()
print(tensor.is_contiguous())
print(tensor)
print(tensor.stride())

# tensor = tensor.view(-1)  # Throw a Runtime error
tensor = tensor.contiguous().view(6, 2)
print(tensor.is_contiguous())
print(tensor)
print(tensor.stride())
