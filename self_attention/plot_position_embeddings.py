import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

d_pos_vec = 512
max_positions = 100
# keep dim 0 for padding token position encoding zero vector
position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(max_positions)])

position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

pos1 = position_enc[1]
pos2 = position_enc[2]
pos10 = position_enc[10]
pos99 = position_enc[99]

print(cosine(pos1, pos2))
print(cosine(pos1, pos10))
print(cosine(pos1, pos99))


n_tokens = 100
x = np.arange(0, n_tokens)
pos_enc = position_enc[x]
print(pos_enc.shape)

plt.figure(figsize=(12, 8))
plt.pcolormesh(pos_enc)
plt.xlabel('Embedding Dimensions')
plt.ylabel('Token Position')
plt.xlim((0, d_pos_vec))
plt.ylim((n_tokens, 0))
plt.colorbar()
plt.show()
