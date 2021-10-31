# Self Attention: Implementation
# Steps
# 1. Create Query, Key, and Value using input vectors
# 2. Compute attention scores using Query and Key (transpose)
# 3. Convert attention scores to probability distribution using softmax
# 4. Compute weighted values by multiplying by multiplying attention scores to corresponding values
# e.g. Q_1*K_1 * V_1, Q_1*K_2 * V_2  ... Q_1*K_N * V_N
#      Q_2*K_1 * V_1, Q_2*K_2 * V_2  ... Q_2*K_N * V_N
#      Q_N*K_1 * V_1, Q_N*K_2 * V_2  ... Q_N*K_N * V_N
# 5. Add-up the weighted values, computed using the scores of a particular query
# e.g. Q_1*K_1 * V_1 + Q_1*K_2 * V_2  ... + Q_1*K_N * V_N (ROW-1, dimension of Values)
#      Q_2*K_1 * V_1 + Q_2*K_2 * V_2  ... + Q_2*K_N * V_N (ROW-2)
#      Q_N*K_1 * V_1 + Q_N*K_2 * V_2  ... + Q_N*K_N * V_N (ROW-3)

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
        attn_scores = matmul(q, k.t())
        print(attn_scores)
        softmax_attn_score = softmax(attn_scores, dim=-1)
        print(numpy.round(softmax_attn_score.detach(), decimals=2))
        v_formatted = v[:, None]
        print(v_formatted)
        softmax_attn_score_transpose = softmax_attn_score.t()
        scores_formatted = softmax_attn_score_transpose[:, :, None]
        print(scores_formatted)
        v_weighted = v_formatted * scores_formatted
        output = v_weighted.sum(dim=0)
        print(numpy.round(output.detach(), decimals=1))
        return output


x = torch.tensor([
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
], dtype=torch.float32)

self_attn = SelfAttention(4, 3)
self_attn(x)
