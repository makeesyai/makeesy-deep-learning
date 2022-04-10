import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.dataset import T_co


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


class TextDatasetIterable(IterableDataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, src_file, tgt_file, vcb_src=None, vcb_tgt=None):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.vcb_src = vcb_src
        self.vcb_tgt = vcb_tgt

    def preprocess(self, zip_line):
        line_src, line_tgt = zip_line
        sample_src = ['<s>'] + line_src.split() + ['</s>']
        sample_tgt = ['<s>'] + line_tgt.split() + ['</s>']

        sample_src_idx = [self.vcb_src.get(t, self.vcb_src.get('<unk>')) for t in sample_src]
        sample_tgt_idx = [self.vcb_tgt.get(t, self.vcb_tgt.get('<unk>')) for t in sample_tgt]
        return (torch.tensor(sample_src_idx, dtype=torch.long),
                torch.tensor(sample_tgt_idx, dtype=torch.long))

    def __iter__(self):
        with open(self.src_file, encoding='utf8') as itr_src, \
                open(self.tgt_file, encoding='utf8') as itr_tgt:
            zip_iter = zip(itr_src, itr_tgt)
            mapped_itr = map(self.preprocess, zip_iter)
            return mapped_itr


class TextDatasetIterableSPM(IterableDataset):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, src_file, tgt_file, spm):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.spm = spm

    def preprocess(self, zip_line):
        line_src, line_tgt = zip_line
        # Encode src and tgt
        sample_src_idx = self.spm.encode(line_src)
        sample_tgt_idx = self.spm.encode(line_tgt)
        return (torch.tensor(sample_src_idx, dtype=torch.long),
                torch.tensor(sample_tgt_idx, dtype=torch.long))

    def __iter__(self):
        with open(self.src_file, encoding='utf8') as itr_src, \
                open(self.tgt_file, encoding='utf8') as itr_tgt:
            for zip_example in zip(itr_src, itr_tgt):
                yield self.preprocess(zip_example)


def generate_batch(data_batch, PAD_IDX):
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
