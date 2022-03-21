import numpy
import torch
from torch import nn, matmul
from torch.nn.functional import softmax

w_query = torch.tensor(
    [
        [0, 0, 1, 0, 0, 1],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
    ], dtype=torch.float32)

w_key = torch.tensor(
    [
        [1, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 1],
    ]
    , dtype=torch.float32)

w_value = torch.tensor(
    [
        [1, 0, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 1],
    ], dtype=torch.float32)


class SelfAttention(nn.Module):
    def __init__(self, embeddings, heads, heads_dim):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.heads_dim =heads_dim
        self.to_query = nn.Linear(embeddings, heads * heads_dim, bias=False)
        self.to_query.weight = nn.Parameter(w_query.t())

        self.to_key = nn.Linear(embeddings, heads * heads_dim, bias=False)
        self.to_key.weight = nn.Parameter(w_key.t())

        self.to_value = nn.Linear(embeddings, heads * heads_dim, bias=False)
        self.to_value.weight = nn.Parameter(w_value.t())

        self.unified_heads = nn.Linear(heads * heads_dim, heads_dim)

    def forward(self, inputs):
        seq_length, emb = inputs.shape
        q_l = self.to_query(inputs)
        # print(q_l)
        q_v = q_l.view(seq_length, self.heads, self.heads_dim)
        # print(q_v)
        q = q_v.transpose(0, 1)
        # print(q)

        k = self.to_key(inputs).view(seq_length, self.heads, self.heads_dim).transpose(0, 1)
        v = self.to_value(inputs).view(seq_length, self.heads, self.heads_dim).transpose(0, 1)
        attn_scores = matmul(q, k.transpose(-2, -1))
        print(attn_scores)
        softmax_attn_score = softmax(attn_scores, dim=-1)

        print(numpy.round(softmax_attn_score.detach(), decimals=2))
        v_formatted = v[:, :, None]
        # print("Value Formatted")
        # print(v_formatted)
        softmax_attn_score_transpose = softmax_attn_score.transpose(-1, -2)
        scores_formatted = softmax_attn_score_transpose[:, :, :,  None]
        # print("Scores formatted")
        # print(scores_formatted)
        v_weighted = v_formatted * scores_formatted
        # print(numpy.round(v_weighted.sum(dim=1).detach(), decimals=2))

        # print(matmul(softmax_attn_score, v))
        print(torch.bmm(softmax_attn_score, v))


x = torch.tensor(
    [
        [1, 0, 1, 0],  # input 1
        [0, 2, 2, 2],  # input 2
        [1, 1, 1, 1],  # input 3
    ], dtype=torch.float32)

self_attn = SelfAttention(4, 2, 3)
self_attn(x)
self_attn.forward_einsum(x.unsqueeze(0))

# x= torch.tensor(
#     [[[0., 1., 1.], [0., 1., 1.]],
#      [[4., 6., 0.], [4., 6., 0.]],
#      [[2., 3., 1.], [2., 3., 1.]]]
# )

