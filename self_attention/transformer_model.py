import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Transformer
from torch.optim import Adam


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
                            self.pos_embedding[:token_embedding.size(0),:])


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


def token2idx(data, vcb):
    data_idx = []
    for sample in data:
        sample_idx = [vcb.get(t, vcb.get('UNK')) for t in sample]
        data_idx.append(sample_idx)
    return data_idx


def load_vocab(file_path):
    word2idx = {'UNK': 0, '<s>': 1, '</s>': 2}
    index = len(word2idx)
    with open(file_path, encoding='utf8') as fin:
        for line in fin:
            word, freq = line.split('\t')
            if word not in word2idx:
                word2idx[word] = index
                index += 1
    return word2idx


def load_data(file_path):
    dada = []
    with open(file_path, encoding='utf8') as fin:
        for line in fin:
            dada.append(['<s>'] + line.split() + ['</s>'])
    return dada


src_vcb = load_vocab('../data/vocab_src.txt')
tgt_vcb = load_vocab('../data/vocab_tgt.txt')
# print(src_vcb)
# print(tgt_vcb)

src_data = load_data('../data/sources.txt')
tgt_data = load_data('../data/targets.txt')

src_data_idx = token2idx(src_data, src_vcb)
tgt_data_idx = token2idx(tgt_data, src_vcb)
# print(src_data_idx)
# print(tgt_data_idx)

model = MyTransformer(len(src_vcb), len(tgt_vcb))
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

count = 0
for example_x, example_y in zip(src_data_idx, tgt_data_idx):
    optimizer.zero_grad()
    example_y_input = example_y[:-1]
    example_y_true = example_y[1:]
    tensor_x = torch.LongTensor([example_x])
    tensor_y_input = torch.LongTensor([example_y_input]).transpose(0, 1)
    tensor_y_true = torch.LongTensor([example_y_true])
    output = model(tensor_x, tensor_y_input)
    if count > 0 and count % 10 == 0:
        print(output.argmax(-1).view(-1).tolist())
        print(example_y_true)
    count += 1
    loss = criterion(output.view(-1, len(tgt_vcb)), tensor_y_true.view(-1))
    loss.backward()
    optimizer.step()
    print(loss.item())

count = 0
with torch.no_grad():
    model.eval()
    for example_x, example_y in zip(src_data_idx, tgt_data_idx):
        example_y_true = example_y[1:]
        tensor_x = torch.LongTensor([example_x])
        memory = model.encode(tensor_x)
        ys = torch.ones(1, 1).fill_(1).long()
        for i in range(100):
            out = model.decode(ys, memory)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).fill_(next_word)], dim=0).long()
            # if next_word == 2:
            #     break
        print(ys.transpose(0, 1))
        count += 1
        if count == 10:
            exit()
