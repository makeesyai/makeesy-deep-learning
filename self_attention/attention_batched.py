# Self Attention: Batched Implementation

# Steps
# 1. Create Query, Key, and Value using input vectors.
# 2. Compute attention scores using Query and Key (transpose).
# 3. Convert attention scores to probability distribution using SoftMax.
# 4. Compute weighted values by multiplying attention scores to corresponding values. 
# [Q₁ x K₁] * V₁, [Q₁ x K₂] * V₂ … [Q₁ x Kₙ] * Vₙ
# [Q₂ x K₁] * V₁, [Q₂ x K₂] * V₂ … [Q₂ x Kₙ] * Vₙ
# …
# [Qₙ x K₁] * V₁, [Qₙ x K₂] * V₂ … [Qₙ x Kₙ] * Vₙ
# 
Where "x" is the dot product and "*" is the pointwise matrix multiplication. Also, Qₙ is defined as-
# Q = [
# [0, 1, 1], # Q₁
# [4, 6, 0], # Q₂
# [2, 3, 1], # Q₃
# ]
# Similarly, Vₙ is a row of Value matrix, and Kₙ is the column of Key Matrix.

# 5. Add-up the weighted values, computed using the scores of a particular query.
# [Q₁ x K₁] * V₁+ [Q₁ x K₂] * V₂ … + [Q₁ x Kₙ] * Vₙ (R₁)
# [Q₂ x K₁] * V₁+ [Q₂ x K₂] * V₂ … + [Q₂ x Kₙ] * Vₙ (R₂)
# …
# [Qₙ x K₁]* V₁+ [Qₙ x K₂]* V₂… + [Qₙ x Kₙ]* Vₙ (Rₙ)


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
