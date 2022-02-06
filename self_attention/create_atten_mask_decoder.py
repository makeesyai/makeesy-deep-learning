# Decoder/Subsequent/Look-ahead Mask: torch.triu()
import torch
pad = 0
trg = torch.tensor([[1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 0, 0]])
trg_mask = (trg == pad).unsqueeze(-2).type(torch.int16)
print(trg_mask)
bs, seq_len = trg.shape
print(bs, seq_len)
mask_shape = (1, seq_len, seq_len)
triu_tensor = torch.triu(torch.ones(mask_shape), diagonal=1).type(torch.int16)
print(triu_tensor)
look_ahead_mask = trg_mask | triu_tensor
print(look_ahead_mask)
# heads = 2
# attn_shape = (bs, heads, seq_len, seq_len)
attn_shape = (bs, seq_len, seq_len)
attn_scores = torch.rand(attn_shape)
print(attn_scores)
attn_scores_mask_subsequent = attn_scores.masked_fill(look_ahead_mask == 1, value=-1e9)
print(attn_scores_mask_subsequent)
