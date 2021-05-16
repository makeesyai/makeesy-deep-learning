# Operations on Tensors: Session 2
# 1. Indexing and Slicing (In slicing the end index is always excluded)

# tensor[d1], tensor[d1, d2], tensor[d1, d2, d3] .. tensor[d1, d2, .. dn]
# tensor[d1_start:d1_end(excluded), d2_start:d2_end, ..dn_start:dn_end)
# tensor[index_d1, index_d2 , dn_start:dn_end)
import numpy
import torch

tensor_1d = torch.tensor([1, 2, 3, 4, 5, 6, 7])
tensor = tensor_1d[4]
print(tensor)
tensor = tensor_1d[2:5]
print(tensor)

tensor_2d = torch.tensor([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10],
                          [11, 12, 13, 14, 15]])

tensor = tensor_2d[2, 3]
print(tensor)
tensor = tensor_2d[1:3, 2:5]
print(tensor)

tensor_3d = torch.tensor([[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]],
                       [[13, 14, 15, 16],
                        [17, 18, 19, 20],
                        [21, 22, 23, 24]],
                       [[25, 26, 27, 28],
                        [29, 30, 31, 32],
                        [33, 34, 35, 36]]
                       ])

tensor = tensor_3d[1:3, 1:3, 2:4]
print(tensor)
tensor = tensor_3d[0, 1, 2:4]
print(tensor)

# 2. Bridge with NumPy
array = numpy.asarray([1, 2, 3, 4, 5])
print(array)
tensor = torch.from_numpy(array)
print(tensor)
numpy_array = tensor.numpy()
print(numpy_array)

