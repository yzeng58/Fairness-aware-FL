import torch
device = torch.device("cuda")
import torch.nn.functional as F

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, *_ = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            _, idx = output.max(dim=1)
            correct += (idx == target).sum().item()

    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

from torch.utils import data
from torchvision import datasets
from torchvision import transforms

train_loader = data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=256, shuffle=True, drop_last=True)

test_loader = data.DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=2048, shuffle=False, drop_last=False)

import torch.nn as nn
from torch.optim import Adam


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        return fc2, [conv1, conv2]

import numpy as np
from large_margin import LargeMarginLoss

lm = LargeMarginLoss(
    gamma=10000,
    alpha_factor=4,
    top_k=1,
    dist_norm=np.inf
)


def train_lm(model, train_loader, optimizer, epoch, lm):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        one_hot = torch.zeros(len(target), 10).scatter_(1, target.unsqueeze(1), 1.).float()
        one_hot = one_hot.cuda()
        optimizer.zero_grad()
        output, feature_maps = model(data)
        #loss = F.mse_loss(output, target) * 5e-4 # l2_loss_weght
        loss = lm(output, one_hot, feature_maps)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

net = Net().to(device)
# net = nn.DataParallel(net).to(device)
optim = Adam(net.parameters())
for i in range(0, 5):
    train_lm(net, train_loader, optim, i, lm)
    test(net, test_loader)