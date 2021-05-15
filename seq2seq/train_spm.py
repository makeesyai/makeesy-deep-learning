import sentencepiece as spm

spm.SentencePieceTrainer.Train(input=['../data/europarl/Europarl.de-en.en',
                                      '../data/europarl/Europarl.de-en.de'],
                               model_prefix='../data/europarl/Europarl.de-en', vocab_size=30000,
                               user_defined_symbols=['<PAD>'], pad_id=3)
