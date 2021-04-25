# Tensor:A tensor is a container which can house data in N dimensions, along with its linear operations,
# though there is difference in what tensors technically are and what we refer to as tensors in practice.

# Types: Float and Int
# Precision
# 64-bit (Float: float64, double-> DoubleTensor + Int: int64, long-> LongTensor)
# 32-bit (Float: float32, float-> FloatTensor + Int: int32, int-> IntTensor)
# 16-bit (Float: float16, half-> HalfTensor + Int, int16, short-> ShortTensor)
# 8-bit (Signed: int8-> ByteTensor + Unsigned: uint8-> CharTensor)

import torch
# print(torch.finfo(torch.float16))
# print(torch.iinfo(torch.int8))

# torch.Tensor() == torch.FloatTensor() vs torch.tensor()
# tensor = torch.Tensor()  # is OK
# print(tensor)
# print(tensor.type())

# tensor = torch.tensor()  # Error
# tensor = torch.tensor([])  # is OK
# print(tensor)
# print(tensor.type())

# tensor = torch.Tensor([1, 2, 4])  # is OK
# print(tensor)
# print(tensor.type())
# tensor = torch.Tensor([1, 2, 4], dtype=torch.float32)  # Not allowed, as it always returns a FloatTensor
# print(tensor)
# print(tensor.type())

# By default, tensor's data type is automatically assigned (the default types: float32 and int64)
# but in case we need to force it, we can use 'dtype' parameter
# tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float16)
# print(tensor)
# print(tensor.type())

# Changing the dtype of a tensor after creation
tensor = torch.randn(5, 3)
print(tensor)
print(tensor.type())
tensor = tensor.type(torch.HalfTensor)
tensor = tensor.type('torch.HalfTensor')  # also valid
print(tensor)
print(tensor.type())

tensor = tensor.double()
print(tensor)
print(tensor.type())
