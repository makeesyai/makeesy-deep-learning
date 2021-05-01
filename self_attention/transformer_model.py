import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerEncoder, Transformer, \
    TransformerDecoderLayer, TransformerEncoderLayer
from torch.optim import Adam


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
        self.generator = nn.Linear(dim_embeddings, tgt_vocab)


    def forward(self, x, y):
        tensor_x = self.src_embeddings(x)
        tensor_y = self.tgt_embeddings(y)
        tensor = self.transformer(tensor_x, tensor_y)
        tensor = self.generator(tensor_x)
        return tensor


def token2idx(data, vcb):
    data_idx = []
    for sample in data:
        sample_idx = [vcb.get(t, vcb.get('UNK')) for t in sample]
        data_idx.append(sample_idx)
    return data_idx


def load_vocab(file_path):
    word2idx = {'UNK': 0}
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
            dada.append(line.split())
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

for example_x, example_y in zip(src_data_idx, tgt_data_idx):
    optimizer.zero_grad()
    tensor_x = torch.LongTensor([example_x])
    tensor_y = torch.LongTensor([example_y])
    output = model(tensor_x, tensor_y)
    loss = criterion(output.view(-1, len(tgt_vcb)), tensor_y.view(-1))
    loss.backward()
    optimizer.step()
    print(loss.item())
