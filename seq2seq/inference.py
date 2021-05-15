import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from seq2seq.data_utils import TextDatasetIterableSPM
from seq2seq.model_utils import load_model
import sentencepiece as spm
from seq2seq.seq2seq_transformer_spm import Seq2SeqTransformer, TokenEmbedding, PositionalEncoding


def generate_batch(data_batch):
    src_batch, tgt_batch = [], []
    for (src_item, tgt_item) in data_batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
src_file = '../data/europarl/source.test.txt'
tgt_file = '../data/europarl/target.test.txt'
sp = spm.SentencePieceProcessor(model_file='../data/wmt/wmt.de-en.model', add_bos=True, add_eos=True)
train_data = TextDatasetIterableSPM(src_file, tgt_file, sp)
PAD_IDX = sp.pad_id()
BOS_IDX = sp.bos_id()
EOS_IDX = sp.eos_id()
num_sps = sp.vocab_size()

BATCH_SIZE = 16
EPOCHS = 16
PATIENCE = 100
train_iter = DataLoader(train_data, batch_size=1,
                        shuffle=False, collate_fn=generate_batch)
count = 0
with torch.no_grad():
    model = load_model('models/pytorch_model.bin')
    model.eval()
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        num_tokens = src.size(0)
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(DEVICE)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).type_as(src.data).fill_(BOS_IDX)
        for i in range(100):
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=-1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        print(f'Translation: {sp.decode(ys.view(-1).tolist())}')
        print(f'Reference: {sp.decode(tgt.view(-1).tolist())}\n')
        count += 1
        if count == 10:
            exit()
