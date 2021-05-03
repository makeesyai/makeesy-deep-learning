import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
import sentencepiece as spm


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
    def __init__(self, src_vocab, tgt_vocab, dim_embeddings=256, n_heads=4, ff_dim=512):
        super(MyTransformer, self).__init__()

        self.src_embeddings = nn.Embedding(src_vocab, dim_embeddings)
        self.tgt_embeddings = nn.Embedding(tgt_vocab, dim_embeddings)
        self.pe = PositionalEncoding(dim_embeddings, dropout=0.01)

        # Encoder model
        encoder_norm = nn.LayerNorm(dim_embeddings)
        enc_layer = nn.TransformerEncoderLayer(dim_embeddings, n_heads, ff_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=6, norm=encoder_norm)

        # Decoder model
        dec_layer = nn.TransformerDecoderLayer(dim_embeddings, n_heads, ff_dim)
        decoder_norm = nn.LayerNorm(dim_embeddings)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=6, norm=decoder_norm)

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


def load_data(file_src, file_tgt, sp):
    dada = []
    with open(file_src, encoding='utf8') as fin_src, \
            open(file_tgt, encoding='utf8') as fin_tgt:
        for line_src, line_tgt in zip(fin_src, fin_tgt):
            # Encode src and tgt
            sample_src_idx = sp.encode(line_src)
            sample_tgt_idx = sp.encode(line_tgt)
            dada.append(
                (torch.tensor(sample_src_idx, dtype=torch.long),
                 torch.tensor(sample_tgt_idx, dtype=torch.long))
            )
    return dada


def generate_batch(data_batch):
    src_batch, tgt_batch = [], []
    for (src_item, tgt_item) in data_batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
src_file = '../data/wmt/WMT-News.de-en.de'
tgt_file = '../data/wmt/WMT-News.de-en.en'
sp = spm.SentencePieceProcessor(model_file='m.model', add_bos=True, add_eos=True)
train_data = load_data(src_file, tgt_file, sp)

PAD_IDX = sp.pad_id()
BOS_IDX = sp.bos_id()
EOS_IDX = sp.eos_id()
num_sps = sp.vocab_size()

BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 100


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)

model = MyTransformer(num_sps, num_sps)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

train = True
if train:
    steps = 0
    total_loss = 0
    for epoch in range(EPOCHS):
        for idx, (src, tgt) in enumerate(train_iter):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:-1, :]
            logits = model(src, tgt_input)
            tgt_out = tgt[1:, :]
            if steps > 0 and steps % PATIENCE == 0:
                print(f'Epoch:{epoch}, Steps: {steps}, Loss:{total_loss/PATIENCE}')
                total_loss = 0
                # print(logits.argmax(-1).view(-1).tolist())
                # print(tgt_out.transpose(0, 1))

            steps += 1
            optimizer.zero_grad()
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

train_iter = DataLoader(train_data, batch_size=1,
                        shuffle=True, collate_fn=generate_batch)
count = 0
with torch.no_grad():
    model.eval()
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        memory = model.encode(src)
        ys = torch.ones(1, 1).type_as(src.data).fill_(BOS_IDX)
        for i in range(100):
            out = model.decode(ys, memory)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        print(ys.transpose(0, 1))
        print(tgt.transpose(0, 1))
        count += 1
        if count == 10:
            exit()
