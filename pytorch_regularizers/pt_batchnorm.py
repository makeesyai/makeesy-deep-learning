# Batch Norm:
# Hard to train deep models with saturating non-linearities
# Batch Norm helps coordinate the update of multiple layers in the model.

import torch
from torch import nn

torch.manual_seed(50)


def batch_norm(batch_x, gamma, beta, eps=1e-5):
    n, d = batch_x.shape

    sample_mean = batch_x.mean(axis=0)
    sample_var = batch_x.var(axis=0, unbiased=False)

    std = torch.sqrt(sample_var + eps)
    x_centered = batch_x - sample_mean

    x_norm = x_centered / std
    out = gamma * x_norm + beta

    cache = (x_norm, x_centered, std, gamma)

    return out, cache


x = torch.rand(2, 3)
print(x)
x_norm, cache = batch_norm(x, gamma=0.02, beta=0.01)
print(x_norm)
print(cache[0])

# With Learnable Parameters
model = nn.BatchNorm1d(3, affine=False, momentum=1, eps=1e-5)
output = model(x)
print(output)
