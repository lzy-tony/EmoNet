import numpy as np
from tqdm import tqdm
from typing import List
from gensim.models.keyedvectors import KeyedVectors

import torch
from torch.utils.data import Dataset, DataLoader

vec_len = 50


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_data():
    model = KeyedVectors.load_word2vec_format("..//data//wiki_word2vec_50.bin", binary=True)
    train_data: List = []
    test_data: List = []
    batch_size = 64
    max_len = 100

    print("==========loading training data==========")
    with open("..//data//train.txt", encoding='utf-8') as train_file:
        for line in tqdm(train_file):
            line = line.split()
            vec = np.zeros([max_len, vec_len], dtype=float)
            for idx in range(1, min(len(line), max_len)):
                if model.has_index_for(line[idx]):
                    vec[idx - 1] = model[line[idx]]
            vec = torch.from_numpy(vec).float()
            label = torch.from_numpy(np.array(int(line[0]))).long()
            train_data.append((vec, label))

    print("==========loading testing data==========")
    with open("..//data//test.txt", encoding='utf-8') as test_file:
        for line in tqdm(test_file):
            line = line.split()
            vec = np.zeros([max_len, vec_len], dtype=float)
            for idx in range(1, len(line)):
                if model.has_index_for(line[idx]):
                    v = model[line[idx]]
                    for j in range(idx, max_len, idx):
                        vec[j - 1] = v
            vec = torch.from_numpy(vec).float()
            label = torch.from_numpy(np.array(int(line[0]))).long()
            test_data.append((vec, label))

    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    load_data()
