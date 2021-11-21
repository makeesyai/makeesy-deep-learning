from torch.functional import einsum
import torch

# NOTATION: "source indices to multiplied-element-wise separated by comma-> target indices"
# Broad-casted element-wise dot product
a = torch.tensor([3])
b = torch.arange(3, 6)
# print(einsum("i,j->j", [a, b]))

# Sum of a tensor
a = torch.arange(3, 6)
# print(einsum("i->", [a]))

# Sum of a matrix rows/columns
a = torch.arange(6).view(2, 3)
# print(einsum("ij->j", [a]))  # the output dimension is column, so will sum columns
# print(einsum("ij->i", [a]))  # the output dimension is row, so will sum rows

# Matrix multiplication
a = torch.arange(6).view(2, 3)
b = torch.arange(12).view(3, 4)
print(a)
print(b)
print(einsum("ik,kj->ij", [a, b]))
