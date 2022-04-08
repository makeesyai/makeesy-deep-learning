# Multiheaded Attention Scaled: Optimized Implementation
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
    def __init__(self, embeddings, heads_dim=512, heads=2):
        super(SelfAttention, self).__init__()

        self.heads = heads
        self.heads_dim = heads_dim

        # In final implementation, we must use bias=True
        self.to_query = nn.Linear(embeddings, heads * heads_dim, bias=False)
        # self.to_query.weight = nn.Parameter(w_query.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_key = nn.Linear(embeddings, heads* heads_dim, bias=False)
        # self.to_key.weight = nn.Parameter(w_key.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_value = nn.Linear(embeddings, heads * heads_dim, bias=False)
        # self.to_value.weight = nn.Parameter(w_value.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.unify_heads = nn.Linear(heads * heads_dim, embeddings, bias=False)
        # self.unify_heads.weight = nn.Parameter(w_unify_heads.t())  # This should be commented in final implementation

    def forward(self, inputs, mask=None):
        # Create Q, K, and V using input vectors
        bs, seq, emb_dim = inputs.shape

        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x num-heads x seq-length x heads_dim
        q = self.to_query(inputs).view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        k = self.to_key(inputs).view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        v = self.to_value(inputs).view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)

        # Scale before Dot-product: q/root_forth(head_dim) and k/root_forth(head_dim)
        q = q/(self.heads_dim ** 1/float(4))
        k = k/(self.heads_dim ** 1/float(4))

        # Compute Attention scores
        attn_scores = matmul(q, k.transpose(-1, -2))
        # Scale after Dot-product : attn_scores/root_square(head_dim)
        # attn_scores = attn_scores/(self.heads_dim ** 1/float(2))

        # Apply masking
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, value=-1e9)

        # Convert attention scores into probability distributions
        softmax_attn_scores = softmax(attn_scores, dim=-1)

        # Compute Weighted Values
        output = matmul(softmax_attn_scores, v)

        # Reshape the weighted values
        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x seq-length x num-heads x heads_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seq, self.heads * self.heads_dim)
        output_final = self.unify_heads(output)
        return output_final


if __name__ == '__main__':
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
    ],
        dtype=torch.float32)

    attn = SelfAttention(embeddings=4, heads_dim=3)
    attn(x)

    # temp = [[[[0., 1., 1.], [0., 1., 1.]],
    #          [[4., 6., 0.], [4., 6., 0.]],
    #          [[2., 3., 1.], [2., 3., 1.]]],
    #
    #         [[[0., 1., 1.], [0., 1., 1.]],
    #          [[4., 6., 0.], [4., 6., 0.]],
    #          [[2., 3., 1.], [2., 3., 1.]]]]
