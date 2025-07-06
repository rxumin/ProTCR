from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from collections import Counter
import json
import pickle
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler



def read_data(file, sep='\t'):
    if file.endswith('csv'):
        df = pd.read_csv(file, sep=sep)
    elif file.endswith('xlsx'):
        df = pd.read_excel(file)
    else:
        raise NotImplementedError(f'does not support file type: {file}')
    peptide = df['peptide'].tolist()
    cdr3 = df['CDR3'].tolist()
    label = None
    if 'label' in df.columns:
        label = df['label'].tolist()
    return peptide, cdr3, label


class PLMDataset(Dataset):
    def __init__(self, peptide, cdr3, labels):
        self.peptide = peptide
        self.cdr3 = cdr3
        self.labels = labels

    def __getitem__(self, index):
        return self.peptide[index], self.cdr3[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 使用 zip(*batch) 将 batch 中的元组解压成三个单独的列表：peptide，cdr3 和 labels
        peptide, cdr3, labels = list(zip(*batch))

        # MHC1
        peptide_max_len = 11  # max([len(x) for x in peptide])
        cdr3_max_len = 24  # max([len(x) for x in cdr3])

        # nettcr
        # peptide_max_len = 9 #max([len(x) for x in peptide])
        # cdr3_max_len = 20 #max([len(x) for x in cdr3])

        # MHC2
        # peptide_max_len = 20  # max([len(x) for x in peptide])
        # cdr3_max_len = 24  # max([len(x) for x in cdr3])
        texts = []
        for i in range(len(peptide)):
            p = list(peptide[i])
            c = list(cdr3[i])
            p = p + ['<pad>'] * (peptide_max_len-len(p)) if len(p) < peptide_max_len else p
            c = c + ['<pad>'] * (cdr3_max_len - len(c)) if len(c) < cdr3_max_len else c
            texts.append(' '.join(p + ['</s>'] + c))
        # 使用 tokenizer对所有文本进行编码
        # 哪些位置是填充的，哪些位置是真实的 token，真实为1，填充为0
        inputs = self.tokenizer.batch_encode_plus(texts, add_special_tokens=True,
                                                  padding="longest", return_tensors='pt')
        # inputs_id为0的位置就是填充的
        # 开始全为0，1表示需要关注的位置，0表示不需要关注的位置
        attention_mask = inputs['attention_mask']
        # 让填充位置不需要关注
        attention_mask[inputs['input_ids']==0] = 0
        inputs['attention_mask'] = attention_mask

        # 初试化全为1的
        token_type_ids = torch.ones(inputs['input_ids'].shape, dtype=torch.int32)
        # token_type_ids (batch_size,seq_len)  peptide为1，cdr3为0,区分肽序列和cdr3序列
        token_type_ids[:, peptide_max_len:] = 0
        inputs['token_type_ids'] = token_type_ids
        inputs['labels'] = torch.LongTensor(labels)
        # print(inputs)
        return inputs



def make_plm_dataloader(train_file, valid_file, test_file, tokenizer,
                        batch_size=128, ):
    train_texts1, train_texts2, train_labels = read_data(train_file)
    valid_texts1, valid_texts2, valid_labels = read_data(valid_file)
    test_texts1, test_texts2, test_labels = read_data(test_file)



    train_dataset = PLMDataset(train_texts1, train_texts2, train_labels)
    valid_dataset = PLMDataset(valid_texts1, valid_texts2, valid_labels)
    test_dataset = PLMDataset(test_texts1, test_texts2, test_labels)
    collator = Collator(tokenizer=tokenizer)

    # 制作dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collator)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    return train_dataloader, valid_dataloader, test_dataloader



