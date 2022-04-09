# EncoderLayer
# 1. Self Attention
# 2. Layer Normalization: What are Pre/Post-Norm?
# 3. FF network with 2 hidden layers: FF + ACT + Dropout + FF
# 4. Dropout Layer


import torch
from torch import nn, matmul, softmax


class MultiheadedAttention(nn.Module):
    def __init__(self, embeddings, heads_dim=512, heads=2):
        super(MultiheadedAttention, self).__init__()

        self.heads = heads

        self.heads_dim = heads_dim

        # In final implementation, we must use bias=True
        self.to_query = nn.Linear(embeddings, heads * self.heads_dim, bias=False)
        # self.to_query.weight = nn.Parameter(w_query.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_key = nn.Linear(embeddings, heads * self.heads_dim, bias=False)
        # self.to_key.weight = nn.Parameter(w_key.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_value = nn.Linear(embeddings, heads * self.heads_dim, bias=False)
        # self.to_value.weight = nn.Parameter(w_value.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.unify_heads = nn.Linear(heads * self.heads_dim, embeddings, bias=False)
        # self.unify_heads.weight = nn.Parameter(w_unify_heads.t())  # This should be commented in final implementation

    def forward(self, inputs, mask=None, kv=None):
        # Create Q, K, and V using input vectors
        bs, seq, emb_dim = inputs.shape

        if kv is not None:
            kv = kv
        else:
            kv = inputs

        kv_bs, kv_seq, kv_emb_dim = kv.size()

        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x num-heads x seq-length x heads_dim
        q = self.to_query(inputs).view(bs, seq, self.heads, self.heads_dim).transpose(1, 2)
        k = self.to_key(kv).view(kv_bs, kv_seq, self.heads, self.heads_dim).transpose(1, 2)
        v = self.to_value(kv).view(kv_bs, kv_seq, self.heads, self.heads_dim).transpose(1, 2)

        # Scale before Dot-product: q/root_forth(head_dim) and k/root_forth(head_dim)
        q = q/(self.heads_dim ** 1 / float(4))
        k = k/(self.heads_dim ** 1 / float(4))

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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_dim, num_heads, ff_size=4, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadedAttention(embeddings=embedding_dim, heads_dim=head_dim, heads=num_heads)
        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_size * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size * embedding_dim, embedding_dim)
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, embeddings, mask=None):
        # Pre-Normalization
        embeddings = embeddings + self.do(self.self_attn(self.attn_norm(embeddings), mask=mask))
        # Pre-Normalization
        embeddings = embeddings + self.do(self.ff(self.final_norm(embeddings)))

        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads=2, num_layers=2, max_seq=512):
        super(TransformerEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.enc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(TransformerEncoderLayer(embedding_dim, head_dim=embedding_dim, num_heads=num_heads))

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids, mask=None):
        embeddings = self.word_embeddings(token_ids)

        hidden_states = []
        for layer in self.enc_layers:
            embeddings = layer(embeddings, mask)
            hidden_states.append(embeddings)

        return hidden_states, self.norm(hidden_states[-1])


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads_dim, num_heads, ff_size=4, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadedAttention(embeddings=embedding_dim, heads_dim=heads_dim, heads=num_heads)
        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.encoder_decoder_attn = MultiheadedAttention(embeddings=embedding_dim, heads_dim=heads_dim, heads=num_heads)
        self.encoder_decoder_attn_norm = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_size * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size * embedding_dim, embedding_dim)
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, embeddings, memory, src_mask=None, trg_mask=None):
        # Pre-Normalization
        embeddings = embeddings + self.do(self.self_attn(self.attn_norm(embeddings), mask=src_mask))

        # print(embeddings.shape)
        # print(memory.shape)

        # Pre-Normalization
        embeddings = embeddings + self.do(
            self.encoder_decoder_attn(self.encoder_decoder_attn_norm(embeddings), mask=trg_mask, kv=memory))

        # Pre-Normalization
        embeddings = embeddings + self.do(self.ff(self.final_norm(embeddings)))

        return embeddings


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads=2, num_layers=2, max_seq=512):
        super(TransformerDecoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.enc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(TransformerDecoderLayer(embedding_dim, heads_dim=embedding_dim, num_heads=num_heads))

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids, memory, src_mask=None, trg_mask=None):
        embeddings = self.word_embeddings(token_ids)

        hidden_states = []
        for layer in self.enc_layers:
            embeddings = layer(embeddings, memory=memory, src_mask=src_mask, trg_mask=trg_mask)
            hidden_states.append(embeddings)

        return hidden_states, self.norm(hidden_states[-1])


if __name__ == '__main__':
    token_ids = torch.randint(low=0, high=511, size=(2, 32))
    print(token_ids)
    model_encoder = TransformerEncoder(vocab_size=512, embedding_dim=768, num_heads=8, num_layers=24)
    model_decoder = TransformerDecoder(vocab_size=512, embedding_dim=768, num_heads=2, num_layers=2)
    hidden_state_enc, output_enc = model_encoder(token_ids)
    # print(len(hidden_state_enc))
    # print(output_enc.shape)

    hidden_state_dec, output_dec = model_decoder(token_ids, output_enc)
    # print(len(hidden_state_dec))
    # print(output_dec.shape)
