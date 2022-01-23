import torch

# torch.triu()
input = torch.tensor([1, 2, 3, 4, 5, 6, 0, 0])
size = input.shape[0]
attn_shape = (1, size, size)
print(torch.triu(torch.ones(attn_shape), diagonal=1))
