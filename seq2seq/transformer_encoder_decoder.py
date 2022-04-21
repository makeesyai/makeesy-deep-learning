import torch
from torch import nn

from seq2seq.seq2seq_model import TokenEmbedding, PositionalEncoding
from seq2seq.transformer_decoder import TransformerDecoder
from seq2seq.transformer_encoder import TransformerEncoder


class Generator(nn.Module):
    def __init__(self, d_model, trg_vocab):
        super(Generator, self).__init__()
        self.generator = nn.Linear(d_model, trg_vocab)

    def forward(self, trg_embeddings):
        return self.generator(trg_embeddings)


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, num_enc_layers, num_dec_layers, num_heads, dropout):
        super(TransformerEncoderDecoder, self).__init__()
        self.src_embeddings = TokenEmbedding(src_vocab, d_model)
        self.trg_embeddings = TokenEmbedding(trg_vocab, d_model)
        self.pos_embeddings = PositionalEncoding(d_model, dropout, maxlen=512)

        self.encoder = TransformerEncoder(d_model, num_heads, num_enc_layers)
        self.decoder = TransformerDecoder(d_model, num_heads, num_dec_layers)

        self.generator = Generator(d_model, trg_vocab)

    def encode(self, token_ids_src, src_mask=None):
        emb = self.src_embeddings(token_ids_src)
        emb = self.pos_embeddings(emb)
        return self.encoder(emb, src_mask)

    def decode(self, token_ids_trg, memory, src_mask=None, trg_mask=None):
        emb = self.trg_embeddings(token_ids_trg)
        emb = self.pos_embeddings(emb)
        return self.decoder(emb, memory, src_mask, trg_mask)

    def generate(self, emb):
        return self.generator(emb)

    def forward(self, src_ids, trg_ids, src_mask=None, trg_mask=None):
        h, enc = self.encode(src_ids, src_mask)
        h, dec = self.decode(trg_ids, enc, src_mask, trg_mask)
        return self.generate(dec)


if __name__ == '__main__':
    src_vocab = 1000
    trg_vocab = 2000
    d_model = 512
    src_token_ids = torch.randint(low=0, high=1000, size=(2, 32))
    trg_token_ids = torch.randint(low=0, high=2000, size=(2, 80))
    print(src_token_ids)
    model = TransformerEncoderDecoder(src_vocab, trg_vocab, d_model, num_enc_layers=2,
                                      num_dec_layers=2, num_heads=4, dropout=0.1)

    logits = model(src_token_ids, trg_token_ids)
    print(logits.shape)
