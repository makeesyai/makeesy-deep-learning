import torch
from torch.utils.data import DataLoader

from seq2seq.data_utils import TextDatasetIterableSPM
from seq2seq.model_utils import load_model
import sentencepiece as spm
from seq2seq.seq2seq_transformer_spm import Seq2SeqTransformer, TokenEmbedding, PositionalEncoding, generate_batch, generate_square_subsequent_mask

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
src_file = '../data/wmt/WMT-News.de-en.de'
tgt_file = '../data/wmt/WMT-News.de-en.en'
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
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE)
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
        print(ys.transpose(0, 1))
        print(tgt.transpose(0, 1))
        count += 1
        if count == 10:
            exit()
