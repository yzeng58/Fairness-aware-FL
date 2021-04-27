import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from numpy import genfromtxt
from Adult_dataset import MyDataset
from alpha_maximization import compute_alpha_with_fairness, compute_alpha_no_fairness
import kernel
batch_size = 20000
learning_rate = 0.1
num_epochs = 10
filename = '/home/wd005/adult_norm.csv'
numpy_data = genfromtxt(filename, delimiter=',')
numpy_data = numpy_data[np.argsort(numpy_data[:,5])]


# i, j = 0, len(numpy_data) - 1
# while i <= j:
#     temp = np.copy(numpy_data[i])
#     numpy_data[i] = numpy_data[j]
#     numpy_data[j] = temp
#     i = i + 1
#     j = j - 1

# split_ratio = 0.6
# male = int(batch_size*split_ratio)
# female = int(batch_size*(1 - split_ratio))
# train = []
# train.extend(numpy_data[0:female])
# train.extend(numpy_data[len(numpy_data) - male: len(numpy_data)])
# train.extend(numpy_data[female: len(numpy_data) - male])
# numpy_data = np.asarray(train)


index = batch_size
train_data = numpy_data[:, 1:-1][0:index // 2]
train_label = numpy_data[:, -1][0:index // 2]

test_data = numpy_data[:, 1:-1][index:]
test_label = numpy_data[:, -1][index:]

train_dataset = MyDataset(train_data, train_label)
test_dataset = MyDataset(test_data, test_label)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = len(numpy_data) - batch_size, shuffle = False)



class logsticRegression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(logsticRegression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        Theta_X = out
        out = torch.sigmoid(out)
        return out, Theta_X


def loss_no_fair(output, target):
    loss = torch.mean((-target * torch.log(output)- (1 - target) * torch.log(1 - output)))

    return loss

def loss_with_fair(output, target, Theta_X, Sen, Sen_bar):
    pred_loss = torch.mean((-target * torch.log(output)- (1 - target) * torch.log(1 - output)))
    fair_loss = torch.mul(Sen - Sen_bar, Theta_X)
    #fair_loss = torch.mean(torch.mul(fair_loss, fair_loss))
    fair_loss = torch.mean(fair_loss)
    print(pred_loss)
    print(fair_loss)
    return pred_loss

def agnostic_loss_no_fair(output, target, weight):

    pred_loss = (-target * torch.log(output)- (1 - target) * torch.log(1 - output))
    pred_loss = torch.mean(torch.mul(pred_loss, weight))
    return pred_loss


def loss_with_agnostic_fair(output, target, Theta_X, weight, Sen, Sen_bar):
    pred_loss = (-target * torch.log(output)- (1 - target) * torch.log(1 - output))
    pred_loss = torch.mean(torch.mul(pred_loss, weight))

    fair_loss = torch.mul(torch.mul(weight, Sen) - Sen_bar, Theta_X)
    fair_loss = torch.mean(torch.mul(fair_loss, fair_loss))

    return pred_loss + 2.0*fair_loss


input_dim = 40
output_dim = 1

model = logsticRegression(input_dim, output_dim)
use_gpu = torch.cuda.is_available()
# if use_gpu:
#     model = model.cuda()



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def compute_model_parameter_fairness(weight, Sen, ratio, option):
    for epoch in range(num_epochs):
        print('*' * 10)
        print(f'epoch {epoch+1}')
        running_acc = 0.0
        model.train()
        for i,  (data, target) in enumerate(train_loader):
            img, label = data, target

            # if use_gpu:
            #     img = img.cuda()
            #     label = label.cuda()
            label = label.view(len(label), -1)

            out, Theta_X = model(img)

            if option == "no_fair":
                loss = loss_no_fair(out, label)
            elif option == "with_fair":
                loss = loss_with_fair(out, label, Theta_X, Sen, ratio)
            elif option == "agnostic_fair":
                loss = loss_with_agnostic_fair(out, label, Theta_X, weight, Sen, ratio)
            elif option == "agnostic_loss_no_fair":
                loss = agnostic_loss_no_fair(out, label, weight)



            pred = torch.where(out > 0.5, torch.tensor(1.0), torch.tensor(0.0))

            running_acc += torch.sum((pred==label))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        running_acc = running_acc.numpy()
        # for p in model.parameters():
        #     print(p)
        print(f'Finish {epoch + 1} training epoch, Acc: {running_acc / len(train_label):.6f}')
        model.eval()
        theta = model.logstic.weight
        bias = model.logstic.bias

        # print(theta.detach().numpy().transpose().shape)
        # print(bias.detach().numpy())
        running_acc = 0.0
        for data in test_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            with torch.no_grad():
                out, Theta_X = model(img)

            pred = torch.where(out > 0.5, torch.tensor(1.0), torch.tensor(0.0))


            Sen_test = numpy_data[:, 0][index:].reshape(-1, 1)

            Sen_test = torch.from_numpy(Sen_test).float()
            # count_S1 = torch.sum(Sen_test)
            # count_S0 = torch.sum(1 - Sen_test)
            count_Y1_S1 = torch.sum(torch.mul(pred, Sen_test))
            count_Y0_S1 = torch.sum(torch.mul(1.0 - pred, Sen_test))
            count_Y1_S0 = torch.sum(torch.mul(pred, 1.0 - Sen_test))
            count_Y0_S0 = torch.sum(torch.mul(1.0 - pred, 1.0 - Sen_test))

            label = label.view(len(label), -1)
            running_acc += torch.sum((pred == label))
            r11 = count_Y1_S1 / len(Sen_test)
            r01 = count_Y0_S1 / len(Sen_test)
            r10 = count_Y1_S0 / len(Sen_test)
            r00 = count_Y0_S0 / len(Sen_test)
            risk_difference = abs(r11 / (r11 + r01) - r10 / (r10 + r00))

        running_acc = running_acc.numpy()
        print(f'Finish {epoch + 1} testing epoch, Acc: {running_acc / len(test_label):.6f}')
        print(f'Finish {epoch + 1} risk difference, risk difference: {risk_difference:.6f}')

    return theta.detach().numpy().transpose(), bias.detach().numpy()

Iteration = 20
sigma = 2
B = 5
Xtr = train_data
Ytr = train_label
ref_idx = np.random.randint(len(test_data), size = 500)
Xref = test_data[ref_idx, :]
Sen = numpy_data[:, 0][0:index].reshape(-1, 1)
Sen_torch = torch.from_numpy(Sen).float()

weight = torch.ones(batch_size, 1)
ratio = 0.1
options = ["no_fair", "with_fair", "agnostic_fair", "agnostic_loss_no_fair"]
ratio_fair = np.sum(Sen) / len(Sen)
print(ratio_fair)
tau = 10
for i in range(Iteration):
    theta, bias = compute_model_parameter_fairness(weight, Sen_torch, ratio_fair, options[0])
    #weight, ratio = compute_alpha_no_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B)
    #weight, ratio = compute_alpha_with_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B, tau)
    #print(weight[0:10])
    #weight = torch.from_numpy(weight).float()


torch.save(model.state_dict(), './logstic.pth')
#

