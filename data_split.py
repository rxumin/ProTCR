import pandas as pd
import re
df1 = pd.read_csv('./data/data.csv')
m1 = df1['peptide'].map(lambda x: len(x)).max()
print(m1)
m1 = df1['CDR3'].map(lambda x: len(x)).max()
print(m1)

df1 = df1.sample(frac=1.)
test_size = 0.2
train_nums = int(df1.shape[0]*(1-test_size))
train_data = df1.iloc[:train_nums]
valid_data = df1.iloc[train_nums:]
train_data.to_csv('./data/train.csv', index=False, sep='\t')
valid_data.to_csv('./data/valid.csv', index=False, sep='\t')



