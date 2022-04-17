from torch import nn, matmul, softmax


class MultiheadedAttention(nn.Module):
    def __init__(self, d_model, heads=2):
        super(MultiheadedAttention, self).__init__()

        self.heads = heads

        # The model dimension split into n-heads
        self.heads_dim = int(d_model / heads)

        # In final implementation, we must use bias=True
        self.to_query = nn.Linear(d_model, heads * self.heads_dim, bias=False)
        # self.to_query.weight = nn.Parameter(w_query.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_key = nn.Linear(d_model, heads * self.heads_dim, bias=False)
        # self.to_key.weight = nn.Parameter(w_key.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_value = nn.Linear(d_model, heads * self.heads_dim, bias=False)
        # self.to_value.weight = nn.Parameter(w_value.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.unify_heads = nn.Linear(heads * self.heads_dim, d_model, bias=False)
        # self.unify_heads.weight = nn.Parameter(w_unify_heads.t())  # This should be commented in final implementation

    def forward(self, inputs, mask=None, kv=None):
        # Create Q, K, and V using input vectors
        bs, seq, emb_dim = inputs.shape

        if kv is not None:
            kv = kv
        else:
            kv = inputs

        kv_bs, kv_seq_len, _ = kv.size()

        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x num-heads x seq-length x heads_dim
        q = self.to_query(inputs).view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        k = self.to_key(kv).view(kv_bs, kv_seq_len, self.heads, self.heads_dim).transpose(1, 2)
        v = self.to_value(kv).view(kv_bs, kv_seq_len, self.heads, self.heads_dim).transpose(1, 2)

        # Scale before Dot-product: q/root_forth(head_dim) and k/root_forth(head_dim)
        q = q / (self.heads_dim ** 1 / float(4))
        k = k / (self.heads_dim ** 1 / float(4))

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
