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
    word2idx = {'PAD': 0, 'UNK': 1, '<s>': 2, '</s>': 3}
    index = len(word2idx)
    with open(file_path, encoding='utf8') as fin:
        for line in fin:
            word, freq = line.split('\t')
            if word not in word2idx:
                word2idx[word] = index
                index += 1
    return word2idx


def load_data(file_src, file_tgt, vcb_src, vcb_tgt):
    dada = []
    with open(file_src, encoding='utf8') as fin_src, \
            open(file_tgt, encoding='utf8') as fin_tgt:
        for line_src, line_tgt in zip(fin_src, fin_tgt):

            sample_src = line_src.split()
            sample_tgt = line_tgt.split()

            sample_src_idx = [vcb_src.get(t, vcb_src.get('UNK')) for t in sample_src]
            sample_tgt_idx = [vcb_src.get(t, vcb_tgt.get('UNK')) for t in sample_tgt]

            dada.append(
                (torch.tensor(sample_src_idx, dtype=torch.long),
                 torch.tensor(sample_tgt_idx, dtype=torch.long))
            )
    return dada


def generate_batch(data_batch):
    de_batch, en_batch = [], []

    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))

    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)

    return de_batch, en_batch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


src_vcb = load_vocab('../data/vocab_src.txt')
tgt_vcb = load_vocab('../data/vocab_tgt.txt')
PAD_IDX = src_vcb.get('PAD')
BOS_IDX = src_vcb.get('<s>')
EOS_IDX = src_vcb.get('</s>')
BATCH_SIZE = 16

# print(src_vcb)
# print(tgt_vcb)

train_data = load_data('../data/sources.txt', '../data/targets.txt', src_vcb, tgt_vcb)
# print(src_data_idx)
# print(tgt_data_idx)

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

model = MyTransformer(len(src_vcb), len(tgt_vcb))
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

count = 0
for idx, (src, tgt) in enumerate(train_iter):
    # print(src, tgt)
    tgt_input = tgt[:-1, :]
    # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
    optimizer.zero_grad()
    output = model(src, tgt_input)
    # print(output.size())
    tgt_out = tgt[1:, :]
    # if count > 0 and count % 10 == 0:
        # print(output.argmax(-1).view(-1).tolist())
        # print(tgt_out)
    # count += 1
    loss = criterion(output.view(-1, len(tgt_vcb)), tgt_out.view(-1))
    loss.backward()
    optimizer.step()
    print(loss.item())


train_iter = DataLoader(train_data, batch_size=1,
                        shuffle=True, collate_fn=generate_batch)
count = 0
with torch.no_grad():
    model.eval()
    for idx, (src, tgt) in enumerate(train_iter):
        memory = model.encode(src)
        print(tgt)
        ys = torch.ones(1, 1).fill_(1).long()
        for i in range(100):
            out = model.decode(ys, memory)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).fill_(next_word)], dim=0).long()
            if next_word == 2:
                break
        print(ys.transpose(0, 1))
        count += 1
        if count == 10:
            exit()
