import math

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader
import sentencepiece as spm

from seq2seq.data_utils import TextDatasetIterableSPM
from seq2seq.model_utils import save_model, load_model


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


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


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim_embeddings=512, n_heads=8, ff_dim=512,
                 n_layers=3, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()

        self.emb_dim = dim_embeddings

        self.src_embeddings = TokenEmbedding(src_vocab, dim_embeddings)
        self.tgt_embeddings = TokenEmbedding(tgt_vocab, dim_embeddings)
        self.pe = PositionalEncoding(dim_embeddings, dropout=dropout)

        # Encoder model
        encoder_norm = nn.LayerNorm(dim_embeddings)
        enc_layer = nn.TransformerEncoderLayer(dim_embeddings, n_heads, ff_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=encoder_norm)

        # Decoder model
        dec_layer = nn.TransformerDecoderLayer(dim_embeddings, n_heads, ff_dim)
        decoder_norm = nn.LayerNorm(dim_embeddings)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers, norm=decoder_norm)

        # Generator
        self.generator = nn.Linear(dim_embeddings, tgt_vocab)

    def encode(self, x, src_mask):
        tensor = self.pe(self.src_embeddings(x))
        return self.encoder(tensor, src_mask)

    def decode(self, y, memory, tgt_mask):
        tensor_y = self.pe(self.tgt_embeddings(y))
        return self.decoder(tensor_y, memory, tgt_mask)

    def forward(self, x, y, src_mask, tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask):
        """
        :param x:
        :param y:
        :param src_mask:
        :param tgt_mask:
        :param src_key_padding_mask:
        :param tgt_key_padding_mask:
        :param memory_key_padding_mask:
        :return:
        Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

        [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

        """
        tensor_x = self.pe(self.src_embeddings(x))
        memory = self.encoder(tensor_x, src_mask, src_key_padding_mask)
        tensor_y = self.pe(self.tgt_embeddings(y))
        tensor = self.decoder(tensor_y, memory, tgt_mask, memory_mask=None,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask
                              )
        logits = self.generator(tensor)
        return logits


def generate_batch(data_batch, PAD_IDX=0):
    src_batch, tgt_batch = [], []
    for (src_item, tgt_item) in data_batch:
        src_batch.append(src_item)
        tgt_batch.append(tgt_item)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, DEVICE):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    src_file = '../data/wmt/WMT-News.de-en.de'
    tgt_file = '../data/wmt/WMT-News.de-en.en'
    sp = spm.SentencePieceProcessor(model_file='../data/wmt/wmt.de-en.model', add_bos=True, add_eos=True)
    # train_data = load_data(src_file, tgt_file, sp)
    train_data = TextDatasetIterableSPM(src_file, tgt_file, sp)

    PAD_IDX = sp.pad_id()
    BOS_IDX = sp.bos_id()
    EOS_IDX = sp.eos_id()
    num_sps = sp.vocab_size()

    BATCH_SIZE = 16
    EPOCHS = 16
    PATIENCE = 100

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=generate_batch)

    model = Seq2SeqTransformer(num_sps, num_sps)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    train = True
    if train:
        steps = 0
        total_loss = 0
        for epoch in range(EPOCHS):
            for idx, (src, tgt) in enumerate(train_iter):
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)
                tgt_input = tgt[:-1, :]

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                    create_mask(src, tgt_input, DEVICE)
                # print(src_mask, src_padding_mask)
                # print(tgt_mask, tgt_padding_mask)
                # exit()
                logits = model(src, tgt_input, src_mask, tgt_mask,
                               src_padding_mask, tgt_padding_mask, src_padding_mask)

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

            # Save the model
            save_model(model, 'models/pytorch_model.bin')

    train_iter = DataLoader(train_data, batch_size=16,
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
            print(ys.transpose(0, 1))
            print(tgt.transpose(0, 1))
            count += 1
            if count == 10:
                exit()
