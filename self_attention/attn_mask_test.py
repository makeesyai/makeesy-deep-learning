# Decoder/Subsequent/Look-ahead Mask: torch.triu()
import torch

from self_attention.multiheaded_attention_scaled import SelfAttention

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
emb_dim = 4
in_shape = (bs, seq_len, emb_dim)
x = attn_score = torch.rand(in_shape)
attn_model = SelfAttention(embeddings=emb_dim, heads_dim=3, heads=2)
attn_model(x, subsequent_mask)
