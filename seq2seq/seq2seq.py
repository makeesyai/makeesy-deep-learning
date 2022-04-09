# EncoderLayer
# 1. Self Attention
# 2. Layer Normalization: What are Pre/Post-Norm?
# 3. FF network with 2 hidden layers: FF + ACT + Dropout + FF
# 4. Dropout Layer


import torch
from torch import nn

from self_attention.multiheaded_attention_scaled import SelfAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, head_dim, num_heads, ff_size=4, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttention(embeddings=embedding_dim, heads_dim=head_dim, heads=num_heads)
        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_size * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size * embedding_dim, embedding_dim)
        )

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.do = nn.Dropout(dropout)

    def forward(self, embeddings):
        # Pre-Normalization
        embeddings = embeddings + self.do(self.self_attn(self.attn_norm(embeddings)))
        # Pre-Normalization
        embeddings = embeddings + self.do(self.self_attn(self.final_norm(embeddings)))

        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads=2, num_layers=2, max_seq=512):
        super(TransformerEncoder, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.enc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(TransformerEncoderLayer(embedding_dim, head_dim=embedding_dim, num_heads=num_heads))

    def forward(self, token_ids):
        embeddings = self.word_embeddings(token_ids)

        hidden_states = []
        for layer in self.enc_layers:
            hidden_states.append(layer(embeddings))

        return hidden_states, hidden_states[-1]


if __name__ == '__main__':
    token_ids = torch.randint(low=0, high=511, size=(2, 32))
    print(token_ids)
    model = TransformerEncoder(vocab_size=512, embedding_dim=768, num_heads=8, num_layers=24)
    hidden_state, output = model(token_ids)
    print(len(hidden_state))
    print(output.shape)
