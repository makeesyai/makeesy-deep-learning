# Decoder/Subsequent/Look-ahead Mask: torch.triu()
import torch
pad = 0
trg = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 0, 0]])
bs, seq_len = trg.shape
mask_shape = (1, seq_len, seq_len)
ones_tensor = torch.ones(mask_shape)
print(ones_tensor)
triu_tesnor = torch.triu(ones_tensor, diagonal=1).type(torch.int16)
trg_mask = (trg == pad).type(torch.int16).unsqueeze(-2)
subsequent_mask = triu_tesnor | trg_mask
print(trg_mask)
print(triu_tesnor)
print(subsequent_mask)
# Multi head
heads = 2
attn_shape = (bs, heads, seq_len, seq_len)
# For Single Head
# attn_shape = (bs, seq_len, seq_len)
attn_score = torch.rand(attn_shape)
attn_score_masked = attn_score.masked_fill(subsequent_mask == 1, value=-1e9)
print(attn_score_masked)
