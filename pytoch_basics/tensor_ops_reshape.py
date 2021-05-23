# modification of the shape of the tensor
# 1. (un)squeeze tensor
import torch

y = torch.tensor([[[1, 2, 3, 4]]])
print(y.shape)
y = y.squeeze(0)
print(y.shape)

y = torch.tensor([1, 2, 3, 4])
print(y.shape)
y = y.unsqueeze(1)
print(y.shape)

# 2. transpose tensor (tensor.t() vs tensor.transpose())
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
print(x.shape)
print(x.t().shape)
x = torch.tensor([[[1, 2, 3, 4],
                  [5, 6, 7, 8]],
                  [[9, 10 , 11, 12],
                   [13, 14, 15, 16]],
                  [[9, 10, 11, 12],
                   [13, 14, 15, 16]]
                  ])
print(x.shape)
x = x.transpose(0, 2)
print(x.shape)

# 3. view tensor
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
print(x.shape)
x = x.t()
x = x.view(1, 8)
print(x.shape)
print(x.is_contiguous())

# 4. reshape tensor
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
print(x.shape)
x = x.reshape(8, 1)
print(x.shape)
print(x.is_contiguous())

# Just want to reshape tensors, use torch.reshape, If you're also concerned about memory usage and want to
# ensure that the two tensors share the same data, use torch.view

# contiguous vs not contiguous tensor
# print(y.shape, y.stride())
# print(y.is_contiguous())
