import torch
pad = 0

# Subsequent Mask: torch.triu()
trg = torch.tensor([4, 5, 6, 7, 0, 0])
trg_mask = (trg == pad).unsqueeze(-2).type(torch.int16)
seq_len = trg.shape[-1]
attn_shape = (1, seq_len, seq_len)
look_ahead_mask = trg_mask | torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.int16)
print(look_ahead_mask)
attn_scores = torch.rand(attn_shape)
print(attn_scores)
attn_scores_mask_subsequent = attn_scores.masked_fill(look_ahead_mask == 1, value=-1e9)
print(attn_scores_mask_subsequent)
