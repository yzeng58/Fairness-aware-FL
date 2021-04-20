import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.x = self.dataset.x[self.idxs]
        self.y = self.dataset.y[self.idxs]
        self.sen = self.dataset.sen[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        feature, label, sensitive = self.dataset[self.idxs[item]]
        return feature, label, sensitive
        # return self.x[item], self.y[item], self.sen[item]
    
class logReg(torch.nn.Module):
    """
    Logistic regression model.
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x.float())
        probas = torch.sigmoid(logits)
        return probas.type(torch.FloatTensor), logits
    
def RD(n_yz):
    """
    Given a dictionary of number of samples in different groups, compute the risk difference.
    """
    return abs(n_yz[(1,1)]/(n_yz[(1,1)] + n_yz[(0,1)]) - n_yz[(1,0)]/(n_yz[(0,0)] + n_yz[(1,0)]))

def loss_func(option, logits, targets, distance, sensitive, mean_sensitive, larg = 1):
    """
    Loss function. 
    """
    acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
    fair_loss = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), distance.T[0])
    fair_loss = torch.mean(torch.mul(fair_loss, fair_loss)) # modified mean to sum
    if option == 'Zafar':
        return acc_loss + larg*fair_loss, acc_loss, larg*fair_loss
    else:
        return acc_loss, acc_loss, larg*fair_loss