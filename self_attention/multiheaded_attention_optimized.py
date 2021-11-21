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
    def __init__(self, embeddings, heads, heads_dim):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.heads_dim =heads_dim
        self.to_query = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_query.weight = nn.Parameter(w_query.t())

        self.to_key = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_key.weight = nn.Parameter(w_key.t())

        self.to_value = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_value.weight = nn.Parameter(w_value.t())

        self.unify_heads = nn.Linear(heads * heads_dim, heads_dim, bias=False)
        self.unify_heads.weight = nn.Parameter(w_unify_heads.t())

    def forward(self, inputs):
        bs, seq, emb = inputs.shape
        q = self.to_query(inputs)
        q_t = q.view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        k = self.to_key(inputs)
        k_t = k.view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        v = self.to_value(inputs)
        v_t = v.view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        attn_scores_t = matmul(q_t, k_t.transpose(-2, -1))
        softmax_attn_score_t = softmax(attn_scores_t, dim=-1)
        weighted_v = matmul(softmax_attn_score_t, v_t)
        weighted_v = weighted_v.transpose(1, 2)
        output = weighted_v.contiguous().view(bs, seq, self.heads * self.heads_dim)
        print(output)

        output_final = self.unify_heads(output)
        print(output_final)


x = torch.tensor([
    [
        [1, 0, 1, 0],  # input 1
        [0, 2, 2, 2],  # input 2
        [1, 1, 1, 1],  # input 3
    ],
    [
        [1, 0, 1, 0],  # input 1
        [0, 2, 2, 2],  # input 2
        [1, 1, 1, 1],  # input 3
    ]
], dtype=torch.float32)


self_attn = SelfAttention(4, 2, 3)
self_attn(x)

# y = torch.tensor(
#     [
#         [
#             [[0., 1., 1.], [0., 1., 1.]],
#             [[4., 6., 0.], [4., 6., 0.]],
#             [[2., 3., 1.], [2., 3., 1.]]
#         ],
#         [
#             [[0., 1., 1.], [0., 1., 1.]],
#             [[4., 6., 0.], [4., 6., 0.]],
#             [[2., 3., 1.], [2., 3., 1.]]
#         ]
#     ])

