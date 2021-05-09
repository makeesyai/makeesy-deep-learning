# 1. Indexing and Slicing (In slicing the end index is always excluded)
# tensor[d1], tensor[d1, d2], tensor[d1, d2, d3] .. tensor[d1, d2, .. dn]
# tensor[d1_start:d1_end(excluded), d2_start:d2_end, ..dn_start:dn_end)
# tensor[index_d1, index_d2 , dn_start:dn_end)
import numpy
import torch

x = torch.tensor([1, 2, 3, 4, 5, 6])
print(x[4])
print(x[1:3])

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
print(x[2])
print(x[2, 1])
print(x[0:2, 1:2])

x = torch.tensor([[[1, 2, 3],
                   [3, 4, 5],
                   [6, 7, 8]],
                  [[9, 10, 11],
                   [12, 13, 14],
                   [15, 16, 17]]])
print(x[1, 2, 0])

# 2. Bridge with NumPy
x = numpy.array([[1, 2, 3], [4, 5, 6]])
y = torch.from_numpy(x)
print(y)
z = y.numpy()
print(z)
