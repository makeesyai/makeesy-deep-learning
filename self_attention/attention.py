# Self Attention: Implementation

# Steps
# 1. Create Query, Key, and Value using input vectors
# 2. Compute attention scores using Query and Key (transpose)
# 3. Convert attention scores to probability distribution using softmax

# 4. Compute weighted values by multiplying by multiplying attention scores to corresponding values
# Q_1 x K_1 * V_1, Q_1 x K_2 * V_2  ... Q_1 x K_N * V_N
# Q_2 x K_1 * V_1, Q_2 x K_2 * V_2  ... Q_2 x K_N * V_N
# ...
# Q_N x K_1 * V_1, Q_N x K_2 * V_2  ... Q_N x K_N * V_N
# Where "x" is the dot product and  "*" is the point-wise matrix multiplication.
# Also, Q_Ns are defined as-
# Q = [
#     [0, 1, 1],  # Q_1
#     [4, 6, 0],  # Q_2
#     [2, 3, 1],  # Q_3
# ]
# Similarly, V_Ns are the rows of Value matrix, and
# K_Ns are the columns of Key(transpose) Matrix.
# 
# 5. Add-up the weighted values, computed using the scores of a particular query e.g. 
# Q_1 x K_1 * V_1 + Q_1 x K_2 * V_2  ... + Q_1 x K_N * V_N (ROW-1)
# Q_2 x K_1 * V_1 + Q_2 x K_2 * V_2  ... + Q_2 x K_N * V_N (ROW-2)
# ...
# Q_N x K_1 * V_1 + Q_N x K_2 * V_2  ... + Q_N x K_N * V_N (ROW-3)

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
    def __init__(self, emb_dim, model_dim):
        super(SelfAttention, self).__init__()
        self.emb_dim = emb_dim
        self.model_dim = model_dim
        self.to_query = nn.Linear(emb_dim, model_dim, bias=False)
        self.to_query.weight = nn.Parameter(w_query.t())
        self.to_key = nn.Linear(emb_dim, model_dim, bias=False)
        self.to_key.weight = nn.Parameter(w_key.t())
        self.to_value = nn.Linear(emb_dim, model_dim, bias=False)
        self.to_value.weight = nn.Parameter(w_value.t())

    def forward(self, inputs):
        # Create Q, K, and V using input vectors
        q = self.to_query(inputs)
        k = self.to_key(inputs)
        v = self.to_value(inputs)
        # Compute Attention scores
        attn_scores = matmul(q, k.t())
        # Convert attention scores into probability distributions
        softmax_attn_scores = softmax(attn_scores, dim=-1)

        # Format the values
        print(v)
        v_formatted = v[:, None]
        print(v_formatted)

        # Format the attention scores
        print(softmax_attn_scores)
        softmax_attn_scores_transpose = softmax_attn_scores.t()
        attn_scores_formatted = softmax_attn_scores_transpose[:, :, None]
        print(attn_scores_formatted)

        #  Compute the final output
        v_weighted = attn_scores_formatted * v_formatted
        print(v_weighted)
        output = v_weighted.sum(dim=0)
        print(output)
        print(softmax_attn_scores)
        # print(v)
        # print(matmul(softmax_attn_scores, v))


x = torch.tensor([
    [1, 0, 1, 0],  # input 1
    [0, 2, 2, 2],  # input 2
    [1, 1, 1, 1],  # input 3
], dtype=torch.float32)

attn = SelfAttention(4, 3)
attn(x)

final_output = torch.tensor([
    [1.9100, 3.2405, 3.5752],
    [2.0000, 3.9999, 4.0000],
    [2.0000, 3.9865, 3.9932]
])
