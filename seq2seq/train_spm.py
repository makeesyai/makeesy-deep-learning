import sentencepiece as spm

spm.SentencePieceTrainer.Train(input=['../data/wmt/WMT-News.de-en.de',
                                      '../data/wmt/WMT-News.de-en.en'],
                               model_prefix='../data/wmt/wmt.de-en', vocab_size=8000,
                               user_defined_symbols=['<PAD>'], pad_id=3)
