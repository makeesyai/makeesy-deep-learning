import pandas as pd
from sklearn.model_selection import train_test_split

src_file = open('../data/europarl/Europarl.de-en.de', encoding='utf8')
tgt_file = open('../data/europarl/Europarl.de-en.en', encoding='utf8')
zip_lines = zip(src_file, tgt_file)
df = pd.DataFrame(zip_lines, columns=['source', 'target'])
train, test = train_test_split(df, test_size=0.1)
train, dev = train_test_split(train, test_size=0.1)
print(train.shape)
print(test.shape)
print(dev.shape)
