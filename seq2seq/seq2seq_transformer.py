import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader


class SeqEncoder(nn.Module):
    def __init__(self, dim_embeddings=128, n_heads=2, ff_dim=512):
        super(SeqEncoder, self).__init__()
        enc_layer = TransformerEncoderLayer(dim_embeddings, n_heads, ff_dim)
        self.transformer = TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, x):
        tensor = self.transformer(x)
        print(tensor.size())


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


class MyTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim_embeddings=128, n_heads=2, ff_dim=512):
        super(MyTransformer, self).__init__()

        self.src_embeddings = nn.Embedding(src_vocab, dim_embeddings)
        self.tgt_embeddings = nn.Embedding(tgt_vocab, dim_embeddings)
        self.pe = PositionalEncoding(dim_embeddings, dropout=0.01)

        # Encoder model
        encoder_norm = nn.LayerNorm(dim_embeddings)
        enc_layer = nn.TransformerEncoderLayer(dim_embeddings, n_heads, ff_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1, norm=encoder_norm)

        # Decoder model
        dec_layer = nn.TransformerDecoderLayer(dim_embeddings, n_heads, ff_dim)
        decoder_norm = nn.LayerNorm(dim_embeddings)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=1, norm=decoder_norm)

        # Generator
        self.generator = nn.Linear(dim_embeddings, tgt_vocab)

    def encode(self, x):
        memory = self.pe(self.src_embeddings(x))
        return self.encoder(memory)

    def decode(self, y, memory):
        tensor_y = self.pe(self.tgt_embeddings(y))
        return self.decoder(tensor_y, memory)

    def forward(self, x, y):
        memory = self.encode(x)
        tensor = self.decode(y, memory)
        logits = self.generator(tensor)
        return logits


