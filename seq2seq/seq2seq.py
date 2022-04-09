import torch
from torch import nn

from self_attention.multiheaded_attention_scaled import SelfAttention


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, head_dim, num_head, ff_size, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = SelfAttention(embeddings=embedding_dim, heads_dim=head_dim, heads=num_head)
        self.attn_norm = nn.LayerNorm(embedding_dim)

        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_size * embedding_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
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


if __name__ == '__main__':
    embeddings = torch.rand(2, 3, 512)
    model = TransformerLayer(embedding_dim=512, head_dim=128, num_head=2, ff_size=2)
    output = model(embeddings)
    print(output.shape)
