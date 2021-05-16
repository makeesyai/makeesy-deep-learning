from argparse import ArgumentParser

import sentencepiece as spm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_files', type=list,
                        default=['../data/europarl/Europarl.de-en.en', '../data/europarl/Europarl.de-en.de'],
                        help='List of files to be added in training SentencePiece Model.')
    parser.add_argument('--model_prefix',
                        default='../data/europarl/Europarl.de-en',
                        help='The model prefix.')
    parser.add_argument('--vocab_size', default=30000, type=int,
                        help='The max vocab size.')

    args = parser.parse_args()

    spm.SentencePieceTrainer.Train(input=args.input_files,
                                   model_prefix=args.model_prefix, vocab_size=args.vocab_size,
                                   user_defined_symbols=['<PAD>'], pad_id=3)
