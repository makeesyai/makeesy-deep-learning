import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, src_file, tgt_file, vcb_src, vcb_tgt):
        self.data = []
        with open(src_file, encoding='utf8') as fin_src, \
                open(tgt_file, encoding='utf8') as fin_tgt:
            for line_src, line_tgt in zip(fin_src, fin_tgt):
                sample_src = ['<s>'] + line_src.split() + ['</s>']
                sample_tgt = ['<s>'] + line_tgt.split() + ['</s>']

                sample_src_idx = [vcb_src.get(t, vcb_src.get('<unk>')) for t in sample_src]
                sample_tgt_idx = [vcb_tgt.get(t, vcb_tgt.get('<unk>')) for t in sample_tgt]
                self.data.append(
                    (torch.tensor(sample_src_idx, dtype=torch.long),
                     torch.tensor(sample_tgt_idx, dtype=torch.long))
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def generate_batch(data_batch):
    src_batch, tgt_batch = [], []
    for (src_item, tgt_item) in data_batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


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


src_file = '../data/wmt/WMT-News.de-en.de'
tgt_file = '../data/wmt/WMT-News.de-en.en'
max_vocab = 100000
src_vcb = create_vocab(src_file, max_vocab)
tgt_vcb = create_vocab(tgt_file, max_vocab)

PAD_IDX = src_vcb.get('<pad>')

dataset = TextDataset(src_file, tgt_file, src_vcb, tgt_vcb)
data_itr = DataLoader(dataset, batch_size=12, collate_fn=generate_batch)

for batch in data_itr:
    print(batch)
    exit()
