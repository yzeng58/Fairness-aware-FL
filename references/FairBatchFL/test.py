import torch
from torch.utils.data import Dataset, DataLoader
from cvxopt import matrix, solvers
from numpy import genfromtxt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

from numpy import genfromtxt

filename = 'adult_norm.csv'
numpy_data = genfromtxt(filename, delimiter=',')

index = 30000
train_data = numpy_data[:, 1:-1][0:index]
train_label = numpy_data[:, -1][0:index]
test_data = numpy_data[:, 1:-1][index:]
test_label = numpy_data[:, -1][index:]

test_data = test_data[0:500]



def gaussian_kernel(X1, X2, sigma):
    t1= X1.shape[0]
    t2 = X2.shape[0]

    diag1 = np.diag(np.matmul(X1, X1.transpose())).reshape(t1, 1)
    diag2 = np.diag(np.matmul(X2, X2.transpose())).transpose().reshape(1, t2)

    D1 = np.tile(diag1, t2)
    D2 = np.tile(diag2, t1).reshape(D1.shape[0], D1.shape[1])
    D3 = 2*np.matmul(X1, X2.transpose())
    D = D1 + D2 - D3
    K = np.exp(-0.5*D/sigma)

    return K

import cvxpy as cp
import numpy as np

# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(n, m)

count = 1

def test_count():
    global count
    count = 3



for i in range(5):
    test_count()
    print('done')
    print(count)