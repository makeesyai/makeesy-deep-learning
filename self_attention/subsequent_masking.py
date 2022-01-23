import torch
from torch import masked_fill
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


scores = torch.tensor([[0.2, 0.3, 0.3, 0.1, 0.1, 0, 0], [0.2, 0.3, 0.3, 0.1, 0.1, 0.2, 0]])
target = torch.tensor([[48, 45, 67, 36, 49, 0, 0], [48, 45, 67, 36, 49, 23, 0]])
print("Target: ", target)
mask = make_std_mask(target, 0)
print("Attn Mask: ", mask)
print(scores)
print("Attn Scores: ", scores)
print("After applying attn mask:")
# scores = scores.masked_fill(mask == 0, -1e9)
mask = mask.unsqueeze(1)
print("Attn Mask: ", mask)

scores = masked_fill(scores, mask == 0, -1e9)
print(scores)
