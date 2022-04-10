import math

import torch
from torch import nn, matmul, softmax, Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


class MultiheadedAttention(nn.Module):
    def __init__(self, d_model, heads=2):
        super(MultiheadedAttention, self).__init__()

        self.d_model = d_model
        self.heads = heads
        self.heads_dim = int(d_model / heads)

        # In final implementation, we must use bias=True
        self.to_query = nn.Linear(self.d_model, heads * self.heads_dim, bias=False)
        # self.to_query.weight = nn.Parameter(w_query.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_key = nn.Linear(self.d_model, heads * self.heads_dim, bias=False)
        # self.to_key.weight = nn.Parameter(w_key.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.to_value = nn.Linear(self.d_model, heads * self.heads_dim, bias=False)
        # self.to_value.weight = nn.Parameter(w_value.t())  # This should be commented in final implementation

        # In final implementation, we must use bias=True
        self.unify_heads = nn.Linear(heads * self.heads_dim, self.d_model, bias=False)
        # self.unify_heads.weight = nn.Parameter(w_unify_heads.t())  # This should be commented in final implementation

    def forward(self, inputs, mask=None, kv=None):
        # Create Q, K, and V using input vectors
        bs, seq, emb_dim = inputs.shape

        if kv is not None:
            kv = kv
        else:
            kv = inputs

        kv_bs, kv_seq, _ = kv.size()

        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x num-heads x seq-length x heads_dim
        q = self.to_query(inputs).view(bs, -1, self.heads, self.heads_dim).transpose(1, 2)
        k = self.to_key(kv).view(kv_bs, -1, self.heads, self.heads_dim).transpose(1, 2)
        v = self.to_value(kv).view(kv_bs, -1, self.heads, self.heads_dim).transpose(1, 2)

        # Scale before Dot-product: q/root_forth(head_dim) and k/root_forth(head_dim)
        q = q / (self.heads_dim ** (1 / 4))
        k = k / (self.heads_dim ** (1 / 4))

        # Compute Attention scores
        attn_scores = matmul(q, k.transpose(-1, -2))
        # Scale after Dot-product : attn_scores/root_square(head_dim)
        # attn_scores = attn_scores/(self.heads_dim ** 1/float(2))

        # Apply masking
        # print(attn_scores)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 1, value=-1e9)
        # print(attn_scores)

        # Convert attention scores into probability distributions
        softmax_attn_scores = softmax(attn_scores, dim=-1)

        # Compute Weighted Values
        output = matmul(softmax_attn_scores, v)

        # Reshape the weighted values
        # Transpose: bs x seq-length x num-heads x heads_dim -> bs x seq-length x num-heads x heads_dim)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.heads * self.heads_dim)
        output_final = self.unify_heads(output)
        return output_final


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_size=2096, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadedAttention(d_model=d_model, heads=num_heads)
        self.attn_norm = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, d_model)
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.do = nn.Dropout(dropout)

    def forward(self, embeddings, mask=None):
        # Pre-Normalization
        embeddings = embeddings + self.do(self.self_attn(self.attn_norm(embeddings), mask=mask))
        # Pre-Normalization
        embeddings = embeddings + self.do(self.ff(self.final_norm(embeddings)))

        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads=2, num_layers=2, max_seq=512):
        super(TransformerEncoder, self).__init__()

        self.enc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(TransformerEncoderLayer(d_model, num_heads=num_heads))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, embeddings, mask=None):

        hidden_states = []
        for layer in self.enc_layers:
            embeddings = layer(embeddings, mask)
            hidden_states.append(embeddings)

        return hidden_states, self.norm(hidden_states[-1])


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_size=2096, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadedAttention(d_model=embedding_dim, heads=num_heads)
        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.encoder_decoder_attn = MultiheadedAttention(d_model=embedding_dim, heads=num_heads)
        self.encoder_decoder_attn_norm = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, embedding_dim)
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, embeddings, memory, src_mask=None, trg_mask=None):
        # Pre-Normalization
        embeddings = embeddings + self.do(self.self_attn(self.attn_norm(embeddings), mask=trg_mask))

        # print(embeddings.shape)
        # print(memory.shape)

        # Pre-Normalization
        embeddings = embeddings + self.do(
            self.encoder_decoder_attn(self.encoder_decoder_attn_norm(embeddings), mask=src_mask, kv=memory))

        # Pre-Normalization
        embeddings = embeddings + self.do(self.ff(self.final_norm(embeddings)))

        return embeddings


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads=2, num_layers=2):
        super(TransformerDecoder, self).__init__()

        self.enc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(TransformerDecoderLayer(embedding_dim, num_heads=num_heads))

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings, memory, src_mask=None, trg_mask=None):
        hidden_states = []
        for layer in self.enc_layers:
            embeddings = layer(embeddings, memory=memory, src_mask=src_mask, trg_mask=trg_mask)
            hidden_states.append(embeddings)

        return hidden_states, self.norm(hidden_states[-1])


class EncoderDecoder(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model=128, n_heads=2, num_enc_layers=6, num_dec_layers=6, ff_dim=512):
        super(EncoderDecoder, self).__init__()

        self.src_embeddings = TokenEmbedding(src_vocab, d_model)
        self.tgt_embeddings = TokenEmbedding(trg_vocab, d_model)
        self.pe = PositionalEncoding(d_model, dropout=0.01)

        # Encoder model
        self.encoder = TransformerEncoder(d_model=d_model, num_heads=n_heads,
                                          num_layers=num_enc_layers)

        # Decoder model
        self.decoder = TransformerDecoder(embedding_dim=d_model, num_heads=n_heads,
                                          num_layers=num_dec_layers)

        # Generator
        self.generator = nn.Linear(d_model, trg_vocab)

    def encode(self, x, src_mask=None):
        embeddings = self.pe(self.src_embeddings(x))
        return self.encoder(embeddings, src_mask)

    def decode(self, y, memory, src_mask=None, trg_mask=None):
        tensor_y = self.pe(self.tgt_embeddings(y))
        return self.decoder(tensor_y, memory, src_mask, trg_mask)

    def forward(self, x, y, src_mask=None, trg_make=None):
        hidden_states, memory = self.encode(x, src_mask)
        hidden_states, tensor = self.decode(y, memory, src_mask, trg_make)
        logits = self.generator(tensor)
        return logits


if __name__ == '__main__':
    src_vocab = 1000
    trg_vocab = 2000
    d_model = 512
    src_token_ids = torch.randint(low=0, high=1000, size=(2, 32))
    trg_token_ids = torch.randint(low=0, high=2000, size=(2, 80))
    print(src_token_ids)
    model_encoder = TransformerEncoder(d_model=d_model, num_heads=8, num_layers=24)
    model_decoder = TransformerDecoder(embedding_dim=d_model, num_heads=2, num_layers=2)
    hidden_state_enc, output_enc = model_encoder(src_token_ids)
    print(len(hidden_state_enc))
    print(output_enc.shape)

    hidden_state_dec, output_dec = model_decoder(trg_token_ids, output_enc)
    print(len(hidden_state_dec))
    print(output_dec.shape)
