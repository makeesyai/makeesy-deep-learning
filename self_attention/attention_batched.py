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
# Where "x" is the dot product and "*" is the point-wise matrix multiplication.
# Also, Qₙ is defined as-
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
        # Create Q, K, and V using input vectors
        q = self.to_query(inputs)
        k = self.to_key(inputs)
        v = self.to_value(inputs)
        # Compute Attention scores
        attn_scores = matmul(q, k.transpose(-1, -2))
        # Convert attention scores into probability distributions
        softmax_attn_scores = softmax(attn_scores, dim=-1)

        # Format the values
        # print(v)
        # v_formatted = v[:, None]
        # print(v_formatted)

        # Format the attention scores
        # print(softmax_attn_scores)
        # softmax_attn_scores_transpose = softmax_attn_scores.t()
        # attn_scores_formatted = softmax_attn_scores_transpose[:, :, None]
        # print(attn_scores_formatted)

        #  Compute the final output
        # v_weighted = attn_scores_formatted * v_formatted
        # print(v_weighted)
        # output = v_weighted.sum(dim=0)
        # print(output)
        output = matmul(softmax_attn_scores, v)
        print(output)


x = torch.tensor([[
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
],
[
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
]],
dtype=torch.float32)

attn = SelfAttention(4, 3)
attn(x)

# tensor([[1.9100, 3.2405, 3.5752],
#         [2.0000, 3.9999, 4.0000],
#         [2.0000, 3.9865, 3.9932]], grad_fn=<MmBackward>)
