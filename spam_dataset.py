import torch
import torch.utils.data
import numpy as np
import json
from PIL import Image

TRAIN_NUM_RATIO = 0.8

class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, train, path='./raw_data/spambase.data'):
        self.train = train
        self.path = path
        Xs, ys = [], []
        with open(self.path) as inf:
            for line in inf:
                info = line.strip().split(',')
                cur_X = [float(x) for x in info[:54]] + [float(info[54])/float(info[56]), float(info[55])/float(info[56])]
                Xs.append(cur_X)
                ys.append(int(info[-1]))
        Xs, ys = np.array(Xs), np.array(ys)

        if train:
            self.st = 0
            self.N = int(TRAIN_NUM_RATIO * len(Xs))
        else:
            self.st = int(TRAIN_NUM_RATIO * len(Xs))
            self.N = len(Xs) - self.st

        self.Xs = torch.FloatTensor(Xs)
        self.ys = ys
        print (self.N, 'Dataset loaded')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]
