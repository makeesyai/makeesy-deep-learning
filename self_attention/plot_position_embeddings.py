# NOTES:
# Sine is with even positions and Cosine is at odd positions
# Keep dim 0 for padding token position encoding zero vector

# Properties
# Check Relative Position property
# Check Relative Position property after linear transformation
# Plot position embeddings to show that only initial a few dimensions matters

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cosine
from torch import nn


class CheckLinerProperty(nn.Module):
    def __init__(self, in_features, out_features):
        super(CheckLinerProperty, self).__init__()
        self.model = nn.Linear(in_features, out_features)

    def forward(self, inp):
        return self.model(torch.tensor(inp, dtype=torch.float32)).detach()


d_model = 512
max_positions = 100

position_enc = np.array([
    [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
    if (pos != 0) else np.zeros(d_model) for pos in range(0, max_positions)])

position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # 2i
position_enc[1:, 1::2] = np.sin(position_enc[1:, 1::2])  # 2i + 1

x = np.arange(0, max_positions)
emb = position_enc[x]

# Check Relative Position property
pos1 = position_enc[1]
pos2 = position_enc[2]
pos10 = position_enc[10]

print(cosine(pos1, pos2))
print(cosine(pos1, pos10))

# Check Relative Position property after linear transformation
lm = CheckLinerProperty(in_features=512, out_features=512)
p1 = lm(pos1)
p2 = lm(pos2)
p10 = lm(pos10)
print(cosine(p1, p2))
print(cosine(p1, p10))

# Plot position embeddings to show that only initial a few dimensions matters
inp_seq = np.arange(0, max_positions)
emb_seq = position_enc[inp_seq]

plt.figure(figsize=(10, 8))
plt.pcolormesh(emb_seq)
plt.xlabel('Position Embeddings')
plt.ylabel('Token Position')
plt.xlim(0, d_model)
plt.ylim(max_positions, 0)
plt.colorbar()
plt.show()
