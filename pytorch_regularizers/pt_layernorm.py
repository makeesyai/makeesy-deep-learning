# Layer Norm:

# Challenges BatchNorm
# 1.
# 2.
# 3

# Example: FC -> LN -> tanh -> FC -> LN -> tanh

import torch
from torch import nn

torch.manual_seed(50)


def layer_norm(batch_x, gamma, beta, eps=1e-5):
    # Manual implementation
    n, d = batch_x.shape

    sample_mean = batch_x.mean(axis=1).view(2, 1)
    sample_var = batch_x.var(axis=1, unbiased=False).view(2, 1)
    std = torch.sqrt(sample_var + eps)
    x_centered = batch_x - sample_mean

    x_norm = x_centered / std
    out = gamma * x_norm + beta

    cache = (x_norm, x_centered, std, gamma)

    return out, cache


x = torch.rand(2, 3)
print(x)
x_norm, cache = layer_norm(x, gamma=0.02, beta=0.01)
print(x_norm)
print(cache[0])

# Pytorch implementation
# With/Without Learnable Parameters
model = nn.LayerNorm(normalized_shape=3)
output = model(x)
print(output)
