# Decoder/Subsequent/Look-ahead Mask: torch.triu()
import torch

from self_attention.multiheaded_attention_scaled import SelfAttention

pad = 0
trg = torch.tensor([[1, 2, 0, 0, 0], [1, 2, 3, 0, 0]])
bs, seq_len = trg.shape
# this will be the same of all samples in a batch and will broadcast
# when we take an OR operation with trg mask
mask_shape = (1, seq_len, seq_len)
ones_tensor = torch.ones(mask_shape)
print(ones_tensor)
triu_tesnor = torch.triu(ones_tensor, diagonal=1).type(torch.int16)
trg_mask = (trg == pad).type(torch.int16).unsqueeze(-2)
subsequent_mask = triu_tesnor | trg_mask  # broadcasting to bs x seq_len x seq_len
print(trg_mask)
print(triu_tesnor)
print(subsequent_mask)
emb_dim=4
inp_shape = (bs, seq_len, emb_dim)
x = torch.rand(inp_shape)
atten_model = SelfAttention(embeddings=emb_dim, heads_dim=3, heads=2)
atten_model(x, subsequent_mask)
