import torch
pad = 0

# Standard Mask
src = torch.tensor([1, 2, 3, 0, 0])
mask = (src == pad).unsqueeze(-2).type(torch.int16)
print(mask)
seq_len = src.shape[-1]
attn_shape = (1, seq_len, seq_len)
print(attn_shape)
attn_scores = torch.rand(attn_shape)
print(attn_scores)
attn_scores_mask_std = attn_scores.masked_fill(mask == 1, value=-1e9)
print(attn_scores_mask_std)
