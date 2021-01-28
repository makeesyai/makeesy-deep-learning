import math

import matplotlib.pyplot as plt
import numpy as np

d_pos_vec = 512
max_positions = 100
# keep dim 0 for padding token position encoding zero vector
position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(max_positions)])

position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

n_tokens = 100
x = np.arange(0, n_tokens, 1)
pos_enc = position_enc[x]
print(pos_enc.shape)

plt.figure(figsize=(12, 8))
plt.pcolormesh(pos_enc, cmap='viridis')
plt.xlabel('Embedding Dimensions')
plt.xlim((0, d_pos_vec))
plt.ylim((n_tokens, 0))
plt.ylabel('Token Position')
plt.colorbar()
plt.show()
