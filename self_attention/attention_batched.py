import numpy
import torch
from torch import nn, matmul
from torch.nn.functional import softmax

w_query = torch.tensor(
    [
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 0],
    ], dtype=torch.float32)

w_key = torch.tensor(
    [
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1],
    ]
    , dtype=torch.float32)

w_value = torch.tensor(
    [
        [1, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=torch.float32)


class SelfAttention(nn.Module):
    def __init__(self, embeddings, model_dim):
        super(SelfAttention, self).__init__()
        self.to_query = nn.Linear(embeddings, model_dim, bias=False)
        self.to_query.weight = nn.Parameter(w_query.t())

        self.to_key = nn.Linear(embeddings, model_dim, bias=False)
        self.to_key.weight = nn.Parameter(w_key.t())

        self.to_value = nn.Linear(embeddings, model_dim, bias=False)
        self.to_value.weight = nn.Parameter(w_value.t())

    def forward(self, inputs):
        q = self.to_query(inputs)
        k = self.to_key(inputs)
        v = self.to_value(inputs)
        attn_scores = matmul(q, k.transpose(-1, -2))
        print(attn_scores)
        softmax_attn_score = softmax(attn_scores, dim=-1)
        print(numpy.round(softmax_attn_score.detach(), decimals=2))
        v_formatted = v[:, :, None]
        print(v_formatted)
        softmax_attn_score_transpose = softmax_attn_score.transpose(-1, -2)
        scores_formatted = softmax_attn_score_transpose[:, :, :, None]
        print(scores_formatted)

        v_weighted = v_formatted * scores_formatted
        print(numpy.round(v_weighted.sum(dim=1).detach(), decimals=2))
        print(softmax_attn_score)
        print(v)
        print(matmul(softmax_attn_score, v))


x = torch.tensor([[
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
], [
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
]], dtype=torch.float32)

self_attn = SelfAttention(4, 3)
self_attn(x)
