import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='m.model', out_type='str',
                                add_bos=True, add_eos=True)

files = ['../data/wmt/WMT-News.de-en.de', '../data/wmt/WMT-News.de-en.en']

for file in files:
    with open(file, encoding='utf8') as fin, \
            open(file+'.sp', 'w', encoding='utf8') as fout:
        for line in fin:
            fout.write(' '.join(sp.Encode(line)) +'\n')
