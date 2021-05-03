import torch

tensor = torch.tensor([1, 2, 2], dtype=torch.float16).to('cuda')
print(tensor)
print(tensor.dtype)
print(tensor.device)

tensor_new = torch.tensor([2,3,4]).type_as(tensor.data)
print(tensor_new)
print(tensor_new.dtype)
print(tensor_new.device)
