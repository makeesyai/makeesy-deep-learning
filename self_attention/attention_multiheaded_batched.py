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
        self.to_query = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_query.weight = nn.Parameter(w_query.t())

        self.to_key = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_key.weight = nn.Parameter(w_key.t())

        self.to_value = nn.Linear(embeddings, heads_dim, bias=False)
        self.to_value.weight = nn.Parameter(w_value.t())

    def forward(self, inputs):
        bs, seq, emb = inputs.shape
        q = self.to_query(inputs)
        # print(q, q.stride())
        q_t = q.view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        # print(q, q.stride())
        # q_t = q.transpose(1, 2)
        # print(q_transpose, q_transpose.stride())
        k = self.to_key(inputs)
        k_t = k.view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        v = self.to_value(inputs)
        v_t = v.view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        print(v_t)
        attn_scores_t = matmul(q_t, k_t.transpose(-2, -1))
        softmax_attn_score_t = softmax(attn_scores_t, dim=-1)
        print(numpy.round(softmax_attn_score_t.detach(), decimals=2))
        weighted_v = matmul(softmax_attn_score_t, v_t)
        print(weighted_v)

    def forward_einsum(self, inputs):
        b = inputs.shape[0]
        q = self.to_query(inputs)
        k = self.to_key(inputs)
        v = self.to_value(inputs)
        q, k, v = map(lambda x: x.reshape(b, -1, self.heads, self.heads_dim), [q, k, v])
        attn_score = torch.einsum("bmhd,bnhd->bhmn", q, k)
        softmax_attn_score = softmax(attn_score, dim=-1)
        output = torch.einsum("bhmn,bnhd->bmhd", softmax_attn_score, v)
        print(output)
        output = output.reshape(b, -1, self.heads * self.heads_dim)
        print(output)
        # print(self.unified_heads(output))


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
self_attn.forward_einsum(x)

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

