import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
max_src_in_batch, max_tgt_in_batch = 0, 0


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


class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data, batch_size, bucket_size=8):
        super().__init__(data)
        self.data = data
        self.batch_size = batch_size
        self.bucket_size = bucket_size

    def batch_size_fn(self, new, count):
        "Keep augmenting batch and calculate total number of tokens + padding."
        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
            max_tgt_in_batch = 0
        max_src_in_batch = max(max_src_in_batch, new[1][0])
        max_tgt_in_batch = max(max_tgt_in_batch, new[1][1])
        src_elements = count * max_src_in_batch
        tgt_elements = count * max_tgt_in_batch
        return max(src_elements, tgt_elements)

    def __iter__(self):
        indices = [(i, (s[0].shape[0], s[1].shape[0])) for i, s in enumerate(self.data)]
        random.shuffle(indices)
        pooled_indices = []

        # create pool of indices with similar lengths
        for i in range(0, len(indices), self.bucket_size * 100):
            indices_sorted_src = sorted(indices[i:i + self.bucket_size * 100], key=lambda x: x[1][0])
            indices_sorted_tgt = sorted(indices_sorted_src, key=lambda x: x[1][1])
            pooled_indices.extend(indices_sorted_tgt)

        pooled_indices_only = [x[0] for x in pooled_indices]
        # yield indices for current batch
        j = 0
        for i in range(0, len(pooled_indices)):
            sofor = self.batch_size_fn(pooled_indices[i], j)
            if sofor < self.batch_size:
                j += 1
                continue
            # Get the start index of the batch
            start = i - j
            end = j
            j = 0
            yield pooled_indices_only[start:start + end]

    def __len__(self):
        return len(self.data)


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size


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
sampler = RandomSampler(train_list, batch_size=4096)
batch_sampler = BatchSampler(sampler, batch_size=4096, drop_last=True)

bucket_dataloader = DataLoader(train_list,
                               batch_sampler=batch_sampler,
                               collate_fn=generate_batch)

for batch in bucket_dataloader:
    print(batch[0].shape, batch[1].shape)
    # exit()
