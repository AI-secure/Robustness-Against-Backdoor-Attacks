import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, pretrained=True, gpu=False):
        super(Model, self).__init__()
        self.pretrained = pretrained
        self.gpu = gpu

        self.model = models.resnet18(pretrained=self.pretrained)
        self.output = nn.Linear(1000, 1)

        if gpu:
            self.cuda()

    def unfix_pert(self,):
        del self.fixed_pert

    def fix_pert(self, sigma, hash_num):
        assert not hasattr(self, 'fixed_pert')
        rand = np.random.randint(2**32-1)
        np.random.seed(hash_num)
        self.fixed_pert = torch.FloatTensor(np.random.randn(1,3,224,224)) * sigma
        if self.gpu:
            self.fixed_pert = self.fixed_pert.cuda()
        np.random.seed(rand)

    def forward(self, x):
        if self.gpu:
            x = x.cuda()
        if hasattr(self, 'fixed_pert'):
            x = x + self.fixed_pert

        x = self.model(x)
        x = self.output(x)

        return x

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        label = label.float()
        return F.binary_cross_entropy_with_logits(pred, label)
