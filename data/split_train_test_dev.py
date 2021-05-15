import pandas as pd
from sklearn.model_selection import train_test_split

src_file = open('../data/europarl/Europarl.de-en.de', encoding='utf8')
tgt_file = open('../data/europarl/Europarl.de-en.en', encoding='utf8')
zip_lines = zip(src_file, tgt_file)
df = pd.DataFrame(zip_lines, columns=['source', 'target'])
train, test = train_test_split(df, test_size=0.01)
train, dev = train_test_split(train, test_size=0.001)
print(train.shape)
print(test.shape)
print(dev.shape)
test.to_csv('../data/europarl/Europarl.de-en_test', header=None, index=None, sep='\t', mode='a')
dev.to_csv('../data/europarl/Europarl.de-en_dev', header=None, index=None, sep='\t', mode='a')
train.to_csv('../data/europarl/Europarl.de-en_train', header=None, index=None, sep='\t', mode='a')
