import torch

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

# 32-bit (Float: float32, float-> FloatTensor + Int: int32, int-> IntTensor)
# 64-bit (Float: float64, double-> DoubleTensor + Int: int64, long-> LongTensor)
# 16-bit (Float: float16, half-> HalfTensor + Int, int16, short-> ShortTensor)
# 8-bit (Signed: int8-> ByteTensor + Unsigned: uint8-> CharTensor)

# By default, tensor's data type is automatically assigned
# but in case we need to force it, we can use 'dtype' parameter
# tensor = torch.tensor([1, 2, 3, 4.], dtype=torch.float16)
# print(tensor)
# print(tensor.type())
