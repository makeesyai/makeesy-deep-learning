import pandas as pd
from sklearn.model_selection import train_test_split

src_file = [l.strip() for l in
            open('../data/europarl/Europarl.de-en.de', encoding='utf8').readlines()]
tgt_file = [l.strip() for l in
            open('../data/europarl/Europarl.de-en.en', encoding='utf8').readlines()]

zip_lines = zip(src_file, tgt_file)
df = pd.DataFrame(zip_lines, columns=['en', 'de'])
train, test = train_test_split(df, test_size=0.003)
train, dev = train_test_split(train, test_size=0.001)
print(train.shape)
print(test.shape)
print(dev.shape)

for column in df.columns:
    test[column].to_csv('../data/europarl/' + 'test.' + column,
                        index=None, header=None)
for column in df.columns:
    dev[column].to_csv('../data/europarl/' + 'dev.' + column,
                        index=None, header=None)
for column in df.columns:
    train[column].to_csv('../data/europarl/' + 'train.' + column,
                        index=None, header=None)
