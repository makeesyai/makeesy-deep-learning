import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cosine
from torch import nn


class CheckLinearProperty(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(CheckLinearProperty, self).__init__()
        self.model = nn.Linear(in_feature, out_feature)

    def forward(self, inp):
        return self.model(torch.tensor(inp, dtype=torch.float32)).detach()


d_pos_vec = 512
max_positions = 100
# keep dim 0 for padding token position encoding zero vector
position_enc = np.array([
    [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
    if pos != 0 else np.zeros(d_pos_vec) for pos in range(max_positions)])

position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

# Check Relative Position property
pos1 = position_enc[1]
pos2 = position_enc[2]
pos10 = position_enc[10]
pos99 = position_enc[99]

print(cosine(pos1, pos2))
print(cosine(pos1, pos10))
print(cosine(pos1, pos99))

# Check Relative Position property after liner transformation
lm = CheckLinearProperty(in_feature=512, out_feature=512)
p1 = lm(pos1)
p2 = lm(pos2)
p10 = lm(pos10)
p99 = lm(pos99)

print(cosine(p1, p2))
print(cosine(p1, p10))
print(cosine(p1, p99))

# Plot position embeddings to show that only initial a few dimensions matters
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
