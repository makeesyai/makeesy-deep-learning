# Attributes of a Tensor: shape, datatype, and the device
import numpy
import torch
from torch import from_numpy

print(torch.cuda.is_available())  # True if machine has a GPU and pytorch is installed with CUDA
tensor = torch.rand((2, 3))
tensor = tensor.to('cuda')  # Move tensor to GPU
print(tensor.dtype)
print(tensor.shape)
print(tensor.DEVICE)


# Initializing a Tensor
# 1. Directly from data
x = [[1, 2], [4, 5]]
tensor = torch.tensor(x)
print(tensor)

# 2. From a NumPy array
numpy_array = numpy.array(x)
print(numpy_array)
print(numpy_array.dtype)

tensor_from_np = from_numpy(numpy_array)
tensor_from_np = tensor_from_np.to('cuda')
print(tensor_from_np)

# 3. From another tensor
tensor = torch.zeros_like(tensor_from_np)
tensor = torch.ones_like(tensor_from_np)
tensor = torch.rand_like(tensor_from_np, dtype=torch.float32)
print(tensor)
print(tensor.dtype)
print(tensor.device)

# 4. With random or constant values
shape = (4, 5,)
tensor = torch.rand(shape) # default dtype is float32
print(tensor)
print(tensor.dtype)
tensor = torch.ones(shape)
print(tensor)
print(tensor.dtype)
tensor = torch.zeros(shape)
print(tensor)
print(tensor.dtype)
