import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32*4*4, 512)
        self.output = nn.Linear(512, 1)

        if gpu:
            self.cuda()

    def unfix_pert(self,):
        del self.fixed_pert

    def fix_pert(self, sigma, hash_num):
        assert not hasattr(self, 'fixed_pert')
        rand = np.random.randint(2**32-1)
        np.random.seed(hash_num)
        self.fixed_pert = torch.FloatTensor(1,1,28,28).normal_(0, sigma)
        if self.gpu:
            self.fixed_pert = self.fixed_pert.cuda()
        np.random.seed(rand)

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        B = x.size()[0]

        if hasattr(self, 'fixed_pert'):
            x = x + self.fixed_pert

        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc(x.view(B,32*4*4)))
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        label = label.float()
        return F.binary_cross_entropy_with_logits(pred, label)
