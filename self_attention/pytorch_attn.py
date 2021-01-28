import torch
from torch import nn
from torch.nn.functional import softmax


class SelfAttn(nn.Module):
    def __init__(self, emb_dim=4, num_heads=2, dim_per_head=3):
        super(SelfAttn, self).__init__()
        self.heads = num_heads
        self.dim_heads = dim_per_head
        self.toquery = nn.Linear(emb_dim, num_heads * dim_per_head, bias=False)
        self.toquery.weight = nn.Parameter(torch.tensor(
            [[1, 0, 1, 1, 0, 1],
             [1, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 1],
             [0, 1, 1, 0, 1, 1]
             ], dtype=torch.float32
        ).t())

        self.tokey = nn.Linear(emb_dim, num_heads * dim_per_head, bias=False)
        self.tokey.weight = nn.Parameter(torch.tensor(
            [[0, 0, 1, 0, 0, 1],
             [1, 1, 0, 1, 1, 0],
             [0, 1, 0, 0, 1, 0],
             [1, 1, 0, 1, 1, 0]
             ], dtype=torch.float32
        ).t())

        self.tovalue = nn.Linear(emb_dim, num_heads * dim_per_head, bias=False)
        self.tovalue.weight = nn.Parameter(torch.tensor(
            [[0, 2, 0, 0, 2, 0],
             [0, 3, 0, 0, 3, 0],
             [1, 0, 3, 1, 0, 3],
             [1, 1, 0, 1, 1, 0]
             ], dtype=torch.float32
        ).t())
        self.unifyheads = nn.Linear(num_heads * dim_per_head, emb_dim)

    def forward(self, input):
        seql, emb = input.size()
        q = self.toquery(input).view(seql, self.heads, self.dim_heads).transpose(0, 1)
        k = self.tokey(input).view(seql, self.heads, self.dim_heads).transpose(0, 1)
        v = self.tovalue(input).view(seql, self.heads, self.dim_heads).transpose(0, 1)

        relevance = torch.matmul(q, k.transpose(-2, -1))
        relevance_score = softmax(relevance, dim=-1)
        out = torch.matmul(relevance_score, v)
        print(out)
        # out = out.transpose(0, 1)
        # print(out)
        # out = out.contiguous()
        # print(out)
        # out = out.view(seql, self.heads*self.dim_heads)
        # print(out)
        # final = self.unifyheads(out)
        # print(final)


x = torch.tensor([
    [1, 0, 1, 0],  # Input 1
    [0, 2, 0, 2],  # Input 2
    [1, 1, 1, 1],  # Input 3
    [1, 2, 1, 2],  # Input 4
    [2, 2, 2, 2],  # Input 5
 ], dtype=torch.float32)

attention = SelfAttn()
attention(x)
