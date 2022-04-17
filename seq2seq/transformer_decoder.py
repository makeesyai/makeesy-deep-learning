# EncoderLayer
# 1. Update Self Attention to Accept KEY-VALUE from Encoder
# 2. Copy the code of Encoder and update it adding Cross attention layer
# 3. FF network with 2 hidden layers: FF(UP-Proj) + ACT + Dropout + FF(Down-Proj)
# 4. Dropout Layer

import torch
from torch import nn

from seq2seq.attention import MultiheadedAttention
from seq2seq.transformer_encoder import TransformerEncoder


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_size=2096, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadedAttention(d_model=d_model, heads=num_heads)
        self.attn_norm = nn.LayerNorm(d_model)

        self.encoder_decoder_attn = MultiheadedAttention(d_model=d_model, heads=num_heads)

        self.encoder_decoder_norm = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, d_model)
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.do = nn.Dropout(dropout)

    def forward(self, embeddings, memory, src_mask=None, trg_mask=None):
        # Attn with Pre-Normalization
        embeddings = embeddings + self.do(self.self_attn(self.attn_norm(embeddings), mask=trg_mask))

        # Cross Attn with Pre-Normalization
        embeddings = embeddings + self.do(self.encoder_decoder_attn(self.encoder_decoder_norm(embeddings), kv=memory, mask=src_mask))

        # FeedForward with Pre-Normalization
        embeddings = embeddings + self.do(self.ff(self.final_norm(embeddings)))

        return embeddings


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads=2, num_layers=2):
        super(TransformerDecoder, self).__init__()

        self.enc_layers = nn.ModuleList()
        for i in range(num_layers):
            self.enc_layers.append(TransformerDecoderLayer(d_model, num_heads=num_heads))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, embeddings, memory, src_mask=None, trg_mask=None):

        hidden_states = []
        for layer in self.enc_layers:
            embeddings = layer(embeddings, memory, src_mask, trg_mask)
            hidden_states.append(embeddings)

        return hidden_states, self.norm(hidden_states[-1])


if __name__ == '__main__':
    dim_model = 512
    src_embeddings = torch.rand(2, 32, dim_model)
    model_encoder = TransformerEncoder(d_model=dim_model, num_heads=8, num_layers=6)
    model_decoder = TransformerDecoder(d_model=dim_model, num_heads=8, num_layers=6)
    hidden, output = model_encoder(src_embeddings)
    print(len(hidden))
    print(output.shape)

    trg_embeddings = torch.rand(2, 35, dim_model)
    hidden, output = model_decoder(trg_embeddings, output)
    print(len(hidden))
    print(output.shape)
