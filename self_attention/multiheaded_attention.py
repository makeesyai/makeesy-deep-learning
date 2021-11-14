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
    def __init__(self, embeddings, head_dim):
        super(SelfAttention, self).__init__()
        # Head 1
        self.to_query_h1 = nn.Linear(embeddings, head_dim, bias=False)
        self.to_query_h1.weight = nn.Parameter(w_query.t())

        self.to_key_h1 = nn.Linear(embeddings, head_dim, bias=False)
        self.to_key_h1.weight = nn.Parameter(w_key.t())

        self.to_value_h1 = nn.Linear(embeddings, head_dim, bias=False)
        self.to_value_h1.weight = nn.Parameter(w_value.t())

        # Head 2
        self.to_query_h2 = nn.Linear(embeddings, head_dim, bias=False)
        self.to_query_h2.weight = nn.Parameter(w_query.t())

        self.to_key_h2 = nn.Linear(embeddings, head_dim, bias=False)
        self.to_key_h2.weight = nn.Parameter(w_key.t())

        self.to_value_h2 = nn.Linear(embeddings, head_dim, bias=False)
        self.to_value_h2.weight = nn.Parameter(w_value.t())

        self.unify_heads = nn.Linear(head_dim * 2, head_dim, bias=False)
        self.unify_heads.weight = nn.Parameter(w_unify_heads.t())

    def forward(self, inputs):
        # HEAD 1
        # Create Q, K, and V using input vectors
        q_h1 = self.to_query_h1(inputs)
        k_h1 = self.to_key_h1(inputs)
        v_h1 = self.to_value_h1(inputs)
        # Compute Attention scores
        attn_scores = matmul(q_h1, k_h1.transpose(-1, -2))
        # Convert attention scores into probability distributions
        softmax_attn_scores = softmax(attn_scores, dim=-1)
        output_h1 = matmul(softmax_attn_scores, v_h1)

        # HEAD 2
        # Create Q, K, and V using input vectors
        q_h2 = self.to_query_h2(inputs)
        k_h2 = self.to_key_h2(inputs)
        v_h2 = self.to_value_h2(inputs)
        # Compute Attention scores
        attn_scores = matmul(q_h2, k_h2.transpose(-1, -2))
        # Convert attention scores into probability distributions
        softmax_attn_scores = softmax(attn_scores, dim=-1)
        output_h2 = matmul(softmax_attn_scores, v_h2)

        concatenate_h1_h2 = torch.cat([output_h1, output_h2], dim=2)
        final_out = self.unify_heads(concatenate_h1_h2)
        print(final_out)


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
