import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def create_vocab(file_path, max_vocab):
    word2idx = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    index = len(word2idx)
    with open(file_path, encoding='utf8') as fin:
        for line in fin:
            words = line.split(' ')
            for word in words:
                if index >= max_vocab:
                    return word2idx
                if word not in word2idx:
                    word2idx[word] = index
                    index += 1
    return word2idx


def load_data(file_src, file_tgt, vcb_src, vcb_tgt):
    dada = []
    with open(file_src, encoding='utf8') as fin_src, \
            open(file_tgt, encoding='utf8') as fin_tgt:
        for line_src, line_tgt in zip(fin_src, fin_tgt):

            sample_src = ['<s>'] + line_src.split() + ['</s>']
            sample_tgt = ['<s>'] + line_tgt.split() + ['</s>']

            sample_src_idx = [vcb_src.get(t, vcb_src.get('<unk>')) for t in sample_src]
            sample_tgt_idx = [vcb_tgt.get(t, vcb_tgt.get('<unk>')) for t in sample_tgt]

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


src_file = '../data/wmt/WMT-News.de-en.de'
tgt_file = '../data/wmt/WMT-News.de-en.en'
max_vocab = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
src_vcb = create_vocab(src_file, max_vocab)
tgt_vcb = create_vocab(tgt_file, max_vocab)
PAD_IDX = src_vcb.get('<pad>')
BOS_IDX = src_vcb.get('<s>')
EOS_IDX = src_vcb.get('</s>')
BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 100

train_list = load_data(src_file, tgt_file, src_vcb, tgt_vcb)
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, new[0].shape[0])
    max_tgt_in_batch = max(max_tgt_in_batch, new[1].shape[0] + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def batch_sampler():
    indices = [(i, s[0].shape[0]) for i, s in enumerate(train_list)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))
    print(pooled_indices)

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]


batch_size = 8
batch_max_tokens = 100
bucket_dataloader = DataLoader(train_list,
                               batch_sampler=batch_sampler(),
                               collate_fn=generate_batch)

for batch in bucket_dataloader:
    print(batch)
    exit()
