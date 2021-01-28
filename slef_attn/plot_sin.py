import math

import matplotlib.pyplot as plt
import numpy as np
import torch

d_pos_vec = 512
n_position = 100
# keep dim 0 for padding token position encoding zero vector
position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
x = np.arange(0, n_position)
plt.plot(position_enc, x)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()

# t1 = torch.tensor([1, 2, 3, 4])
# t2 = torch.tensor([1, 2, 3, 4])
#
# print(t1 * t2)
# position = torch.arange(0., 100).unsqueeze(1)
# print(position)
# div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
# print(div_term)
# out = position * div_term
