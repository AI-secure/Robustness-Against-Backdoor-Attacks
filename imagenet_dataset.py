import torch
import torch.utils.data
import numpy as np
import json
from PIL import Image

class DogCatDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform, path='./raw_data/dog_and_cat/train'):
        self.train = train
        self.transform = transform
        self.path = path
        if train:
            self.st = 0
            self.N = 20000
        else:
            self.st = 20000
            self.N = 5000

        self.dataset = []
        for idx in range(self.N):
            n = (idx+self.st) // 2
            lab = (idx+self.st) % 2
            lab_str = 'cat' if lab == 0 else 'dog'
            path = self.path + '/%s.%d.jpg'%(lab_str, n)
            img = Image.open(path).convert("RGB")
            self.dataset.append((self.transform(img), lab))
        print (self.N, 'Dataset loaded')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.dataset[idx]

class DogFishDataset(torch.utils.data.Dataset):
    def __init__(self, train, transform, path='./raw_data/dog_and_fish/train'):
        self.train = train
        self.transform = transform
        self.path = path
        if train:
            self.st = 0
            self.N = 10000
        else:
            self.st = 10000
            self.N = 2000

        self.dataset = []
        for idx in range(self.N):
            n = (idx+self.st) // 2
            lab = (idx+self.st) % 2
            lab_str = 'fish' if lab == 0 else 'dog'
            path = self.path + '/%s.%d.jpg'%(lab_str, n)
            img = Image.open(path,'rb').convert("RGB")
            self.dataset.append((self.transform(img), lab))
        print (self.N, 'Dataset loaded')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.dataset[idx]
