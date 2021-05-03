import torch

shape = (3, 4, 3)
tensor = torch.randn(shape)
print(tensor)
# print(tensor[:, 2:4, -1:])
print(tensor[2:, 3:, 0:2])

import numpy as np

a2 = np.array([[10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29]])

print(a2[1, 2:4])  # [17 18]
# You can also use a slice of length 1 to do
# something similar (slice 1:2 instead of index 1):
print(a2[1:2, 2:4])  # [[17 18]]

a3 = np.array([[[10, 11, 12], [13, 14, 15], [16, 17, 18]],
               [[20, 21, 22], [23, 24, 25], [26, 27, 28]],
               [[30, 31, 32], [33, 34, 35], [36, 37, 38]]])
print(a3[:2, 1:, :2])  # [[ [13 14] [16 17] ]
#  [ [23 24] [26 27] ]]
# This selects:
# planes :2 (the first 2 planes)
# rows 1: (the last 2 rows)
# columns :2 (the first 2 columns)
