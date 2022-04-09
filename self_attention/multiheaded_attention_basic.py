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

w_unify_heads = torch.tensor(
    [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ], dtype=torch.float32)


class SelfAttention(nn.Module):
    def __init__(self, embeddings, heads_dim, heads=2):
        super(SelfAttention, self).__init__()

        # Head 1
        self.to_query_h1 = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_query_h1.weight = nn.Parameter(w_query.t())

        self.to_key_h1 = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_key_h1.weight = nn.Parameter(w_key.t())

        self.to_value_h1 = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_value_h1.weight = nn.Parameter(w_value.t())

        # Head 2
        self.to_query_h2 = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_query_h2.weight = nn.Parameter(w_query.t())

        self.to_key_h2 = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_key_h2.weight = nn.Parameter(w_key.t())

        self.to_value_h2 = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_value_h2.weight = nn.Parameter(w_value.t())

        self.unify_heads = nn.Linear(heads * heads_dim, embeddings, bias=False)
        self.unify_heads.weight = nn.Parameter(w_unify_heads.t())

    def forward(self, inputs):
        # Head 1
        # Create Q, K, and V using input vectors
        q_h1 = self.to_query_h1(inputs)
        k_h1 = self.to_key_h1(inputs)
        v_h1 = self.to_value_h1(inputs)
        # Compute Attention scores
        attn_scores_h1 = matmul(q_h1, k_h1.transpose(-1, -2))

        # Convert attention scores into probability distributions
        softmax_attn_scores_h1 = softmax(attn_scores_h1, dim=-1)
        output_h1 = matmul(softmax_attn_scores_h1, v_h1)
        print(output_h1)

        # Head 2
        # Create Q, K, and V using input vectors
        q_h2 = self.to_query_h2(inputs)
        k_h2 = self.to_key_h2(inputs)
        v_h2 = self.to_value_h2(inputs)
        # Compute Attention scores
        attn_scores_h2 = matmul(q_h2, k_h2.transpose(-1, -2))

        # Convert attention scores into probability distributions
        softmax_attn_scores_h2 = softmax(attn_scores_h2, dim=-1)
        output_h2 = matmul(softmax_attn_scores_h2, v_h2)
        print(output_h2)
        output_h1_h2 = torch.cat([output_h1, output_h2], dim=-1)
        print(output_h1_h2)
        output_final = self.unify_heads(output_h1_h2)
        print(output_final)


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

# tensor([[[ 7.3951, 13.6313, 10.6356],
#          [ 8.0000, 15.9997, 11.9999],
#          [ 7.9931, 15.9594, 11.9796]],
#
#         [[ 7.3951, 13.6313, 10.6356],
#          [ 8.0000, 15.9997, 11.9999],
#          [ 7.9931, 15.9594, 11.9796]]], grad_fn=<UnsafeViewBackward>)
