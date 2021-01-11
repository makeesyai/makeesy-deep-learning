import numpy as np
import torch

nd_arr = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int16)
print(nd_arr)
print(nd_arr.dtype)

# Convert numpy array to torch tensor
# tensor = torch.from_numpy(nd_arr)
# print(tensor)
# print(tensor.type())
# nd_arr[(0, 0)] = 333  # Update numpy array
# print(tensor)
# print(tensor.type())
# tensor[(0, 0)] = 444  # Update tensor
# print(nd_arr)
# print(nd_arr.dtype)
# Modifications to the tensor will be reflected in the numpy array and vice versa.

# Convert torch tensor to numpy
# tensor = torch.tensor([[1, 2, 3], [1, 2, 3]])
# print(tensor)
# print(tensor.type())
# nd_arr = tensor.numpy()
# nd_arr[(0, 0)] = 333  # Update numpy array
# print(tensor)
# print(tensor.type())
# tensor[(0, 0)] = 444  # Update tensor
# print(nd_arr)
# print(nd_arr.dtype)
# Modifications to the tensor will be reflected in the numpy array and vice versa.

# Another way?
tensor = torch.tensor([[1, 2, 3], [1, 2, 3]])
nd_arr = np.array(tensor)  # creates a new array
print(nd_arr)
print(nd_arr.dtype)
tensor[(0, 0)] = 444  # Update tensor
print(nd_arr)
print(nd_arr.dtype)

# Why not this?
# nd_arr = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int16)
# tensor = torch.tensor(nd_arr)  # creates new tensor
# print(tensor)
# print(tensor.type())
# Update numpy array, this will not change the tensor, as tensor is a copy of nd_arr not the same
# nd_arr[(0, 0)] = 200
# print(tensor)
# print(tensor.type())


# Another possible option?
# tensor = torch.Tensor(nd_arr)   # which is an alias of torch.FloatTensor(), dtype is not copied
# print(tensor)
# print(tensor.type())
# nd_arr[(0, 0)] = 999  # Update numpy array
# print(tensor)
# print(tensor.type())
# tensor[(0, 0)] = 111  # Update tensor
# print(nd_arr)
# print(nd_arr.dtype)

# print(np.ndarray((2, 3), dtype=np.int8))
# nd_arr = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int8)
# print(nd_arr)
# print(nd_arr.dtype)
