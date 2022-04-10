import math

import torch
from torch import nn, Tensor


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


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim_embeddings=512, n_heads=8, ff_dim=512,
                 n_layers=3, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()

        self.emb_dim = dim_embeddings

        self.src_embeddings = TokenEmbedding(src_vocab, dim_embeddings)
        self.tgt_embeddings = TokenEmbedding(tgt_vocab, dim_embeddings)
        self.pe = PositionalEncoding(dim_embeddings, dropout=dropout)

        # Encoder model
        encoder_norm = nn.LayerNorm(dim_embeddings)
        enc_layer = nn.TransformerEncoderLayer(dim_embeddings, n_heads, ff_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=encoder_norm)

        # Decoder model
        dec_layer = nn.TransformerDecoderLayer(dim_embeddings, n_heads, ff_dim)
        decoder_norm = nn.LayerNorm(dim_embeddings)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers, norm=decoder_norm)

        # Generator
        self.generator = nn.Linear(dim_embeddings, tgt_vocab)

    def encode(self, x, src_mask):
        tensor = self.pe(self.src_embeddings(x))
        return self.encoder(tensor, src_mask)

    def decode(self, y, memory, tgt_mask):
        tensor_y = self.pe(self.tgt_embeddings(y))
        return self.decoder(tensor_y, memory, tgt_mask)

    def forward(self, x, y, src_mask, tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask):
        """
        :param x:
        :param y:
        :param src_mask:
        :param tgt_mask:
        :param src_key_padding_mask:
        :param tgt_key_padding_mask:
        :param memory_key_padding_mask:
        :return:
        Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

        [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

        """
        tensor_x = self.pe(self.src_embeddings(x))
        memory = self.encoder(tensor_x, src_mask, src_key_padding_mask)
        tensor_y = self.pe(self.tgt_embeddings(y))
        tensor = self.decoder(tensor_y, memory, tgt_mask, memory_mask=None,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask
                              )
        logits = self.generator(tensor)
        return logits
