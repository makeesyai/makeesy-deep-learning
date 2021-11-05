import torch
from torch import einsum

# 'expression-with-axis-ids -> output-ids'
# Transpose
from torch.nn.functional import softmax

a = torch.arange(6).view(2, 3)
# print(a)
# print(einsum("ij->ji", a))

# Sum Rows
# print(einsum("ij->i", a))

# Sum Columns
# print(einsum("ij->j", a))

# Sum
# print(einsum("ij->", a))

# element-wise multiplication
# print(einsum("ij,ij->ij", a, a))

# Dot product
x = torch.arange(3)
y = torch.arange(3, 6)
# print(einsum("i, i->", x, y))

# Outer Product
# print(einsum("i,j->ij", x, y))

# Matrix Multiplication
x = torch.arange(6).view(2, 3)
y = torch.arange(12).view(3, 4)
# print(einsum("ij,jk->ik", x, y))

# Batch Matrix multiplication
x = torch.arange(6).view(1, 2, 3)
y = torch.arange(12).view(2, 3, 2)
# print(einsum("ijk,ikl->ijl", x, y))

# 4-D matrix multiplication
x = torch.arange(64).view(2, 4, 2, 4)
y = torch.arange(96).view(2, 4, 4, 3)
# print(einsum("bhmd,bhdn->bhmn", x, y))

# Dot Product of 4-D tensors with axis rotation
# q = torch.arange(24).view(2, 2, 2, 3)
# k = torch.arange(24, 48).view(2, 2, 2, 3)
b = 1
h = 2
d = 3
q = torch.tensor([[[0., 1., 1., 0., 1., 1.],
                   [4., 6., 0., 4., 6., 0.],
                   [2., 3., 1., 2., 3., 1.]]]).view(b, -1, h, d)
k = torch.tensor([[[1., 1., 1., 1., 1., 1.],
                   [4., 2., 2., 4., 2., 2.],
                   [3., 1., 2., 3., 1., 2.]]]).view(b, -1, h, d)
v = torch.tensor([[[1., 1., 2., 1., 1., 2.],
                   [2., 4., 4., 2., 4., 4.],
                   [2., 2., 3., 2., 2., 3.]]]).view(b, -1, h, d)
print('Query')
print(q)
print('Key')
print(k)
attn_score = einsum("bmhd,bnhd->bhmn", q, k).type(torch.float32)
print(attn_score)
attn_score_probability = softmax(attn_score, dim=-1)
print(attn_score_probability)
output = einsum("bhmn,bnhd->bmhd", attn_score_probability, v)
print(output)
print(output.reshape(b, -1, h*d))
# In the above example, we have m, and n varying size and sum is over axis d (last axis)
