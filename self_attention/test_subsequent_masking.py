import torch
from torch.autograd import Variable
import numpy as np


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


scores = torch.tensor([0.1, 0.3, 0.3, 0.1, 0.1, 0.03, 0.05, 0.02])
target = torch.tensor([48, 45, 67, 36, 49, 34, 23, 32])
print("Target: ", target)
look_ahead_mask = make_std_mask(target, 0)
print("Attn Mask: ", look_ahead_mask)

print("Attn Scores: ", scores)
print("After applying attn mask:")
scores = scores.masked_fill(look_ahead_mask == 0, -1e9)
print(scores.numpy())
