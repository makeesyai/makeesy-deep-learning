import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerEncoder, Transformer, \
    TransformerDecoderLayer, TransformerEncoderLayer


class SeqEncoder(nn.Module):
    def __init__(self, dim_embeddings=128, n_heads=2, ff_dim=512):
        super(SeqEncoder, self).__init__()
        enc_layer = TransformerEncoderLayer(dim_embeddings, n_heads, ff_dim)
        self.transformer = TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, x):
        tensor = self.transformer(x)
        print(tensor.size())


class MyTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim_embeddings=128, n_heads=2, ff_dim=512):
        super(MyTransformer, self).__init__()
        self.src_embeddings = nn.Embedding(src_vocab, dim_embeddings)
        self.tgt_embeddings = nn.Embedding(tgt_vocab, dim_embeddings)
        self.transformer = Transformer(dim_embeddings, n_heads,
                                       dim_feedforward=ff_dim,
                                       num_decoder_layers=1,
                                       num_encoder_layers=1,
                                       )

    def forward(self, x):
        tensor = self.transformer(x, x)
        print(tensor.size())


x = torch.rand((12, 30, 128))
print(x.size())
# exit()
# model = SeqEncoder()
model = MyTransformer()
model(x)
