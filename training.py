import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from seq2seq.seq2seq_model import EncoderDecoder


if __name__ == '__main__':
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


    def load_data(file_src, file_tgt, vcb_src, vcb_tgt, max_samples=None):
        dada = []
        counter = 0
        with open(file_src, encoding='utf8') as fin_src, \
                open(file_tgt, encoding='utf8') as fin_tgt:
            for line_src, line_tgt in zip(fin_src, fin_tgt):
                counter += 1
                if max_samples and counter >= max_samples:
                    break
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
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, tgt_batch


    def create_masks(src, trg):
        bs, seq_len = trg.shape
        # this will be the same of all samples in a batch and will broadcast
        # when we take an OR operation with trg mask
        mask_shape = (1, seq_len, seq_len)
        ones_tensor = torch.ones(mask_shape)
        triu_tesnor = torch.triu(ones_tensor, diagonal=1).type(torch.int16).to(device)
        trg_mask = (trg == PAD_IDX).type(torch.int16).unsqueeze(-2)
        subsequent_mask = triu_tesnor | trg_mask  # broadcasting to bs x seq_len x seq_len
        subsequent_mask = subsequent_mask.unsqueeze(1)

        src_mask = (src == PAD_IDX).type(torch.int16).unsqueeze(-2).unsqueeze(-2)

        return src_mask.to(device), subsequent_mask.to(device)


    src_file = 'data/wmt/WMT-News.de-en.de'
    tgt_file = 'data/wmt/WMT-News.de-en.en'
    #src_file = 'data/copy/sources.txt'
    #tgt_file = 'data/copy/targets.txt'
    max_vocab = 100000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    src_vcb = create_vocab(src_file, max_vocab)
    trg_vcb = create_vocab(tgt_file, max_vocab)
    idx2word_src = {src_vcb[key]:key for key in src_vcb}
    idx2word_trg = {trg_vcb[key]:key for key in trg_vcb}
    PAD_IDX = src_vcb.get('<pad>')  # Same for trg vocab
    BOS_IDX = src_vcb.get('<s>')  # same for trgvocab
    EOS_IDX = src_vcb.get('</s>')  # same for trg vocab
    BATCH_SIZE = 32
    EPOCHS = 10
    PATIENCE = 100

    train_data = load_data(src_file, tgt_file, src_vcb, trg_vcb)

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)

    model = EncoderDecoder(len(src_vcb), len(trg_vcb))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())

    train = True
    if train:
        steps = 0
        total_loss = 0
        for epoch in range(EPOCHS):
            for idx, (src, trg) in enumerate(train_iter):
                src = src.to(device)
                trg = trg.to(device)
                trg_input = trg[:, :-1]
                trg_out = trg[:, 1:]

                src_mask, trg_mask = create_masks(src, trg_input)

                logits = model(src, trg_input, src_mask, trg_mask)

                if steps > 0 and steps % PATIENCE == 0:
                    print(f'Epoch:{epoch}, Steps: {steps}, Loss:{total_loss/PATIENCE}')
                    total_loss = 0
                    # print(logits.argmax(-1).view(-1).tolist())
                    # print(tgt_out.transpose(0, 1))

                steps += 1
                optimizer.zero_grad()
                loss = criterion(logits.reshape(-1, logits.shape[-1]), trg_out.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    train_iter = DataLoader(train_data, batch_size=1,
                            shuffle=True, collate_fn=generate_batch)
    count = 0
    with torch.no_grad():
        model.eval()
        for idx, (src, trg) in enumerate(train_iter):
            src = src.to(device)
            trg = trg.to(device)

            hidden, memory = model.encode(src)
            ys = torch.ones(1, 1).type_as(src.data).fill_(BOS_IDX)
            for i in range(100):
                hidden, out = model.decode(ys, memory)
                prob = model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=-1)
                next_word = next_word.item()
                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
                if next_word == EOS_IDX:
                    break
            print(" ".join([idx2word_trg.get(idx.item()) for idx in ys[0]]))
            print(" ".join([idx2word_trg.get(idx.item()) for idx in trg[0]]))
            count += 1
            if count == 10:
                exit()
