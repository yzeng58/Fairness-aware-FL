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
from alpha_maximization import compute_alpha_with_fairness, compute_alpha_no_fairness, compute_alpha_with_local_fairness
import kernel
from risk_difference_calculation import risk_difference_calculation
batch_size = 30000
learning_rate = 0.1
num_epochs = 10
filename = '/home/wd005/adult_norm.csv'
numpy_data = genfromtxt(filename, delimiter=',')
#numpy_data = numpy_data[np.argsort(numpy_data[:,0])]




train = []
# split_ratio = 0.25
# male = int(batch_size*split_ratio)
# female = int(batch_size*(1 - split_ratio))
# train.extend(numpy_data[0:female])
# train.extend(numpy_data[len(numpy_data) - male: len(numpy_data)])
# train.extend(numpy_data[female: len(numpy_data) - male])

#local_aware dataset, party 1: 50% male and 50% female
#party 2: party1: 70% male and 30% female
# split_ratio1 = 0.25
# split_ratio2 = 0.15
# male1 = int(batch_size*split_ratio1)
# female1 = int(batch_size*split_ratio1)
# male2 = int(batch_size*(0.6 - split_ratio2))
# female2 = int(batch_size*split_ratio2)
# train.extend(numpy_data[0:female1])
# train.extend(numpy_data[len(numpy_data) - male1: len(numpy_data)])
# train.extend(numpy_data[female1: female1 + female2])
# train.extend(numpy_data[len(numpy_data) - male1 - male2: len(numpy_data) - male1])
# train.extend(numpy_data[female1 + female2: len(numpy_data) - male1 - male2])
# numpy_data = np.asarray(train)
# print(np.sum(numpy_data[:, 0])/len(numpy_data))

#local_aware dataset, party 1: 20% male and 80% female
#party 2: party1: 80% male and 20% female
# split_ratio1 = 0.2
# split_ratio2 = 0.8
# batch_size1 = int(batch_size*0.2)
# batch_size2 = int(batch_size*0.8)
# male1 = int(batch_size1*split_ratio1)
# female1 = int(batch_size1*split_ratio2)
# male2 = int(batch_size2*split_ratio2)
# female2 = int(batch_size2*split_ratio1)
# train.extend(numpy_data[0:female1])
# train.extend(numpy_data[len(numpy_data) - male1: len(numpy_data)])
# train.extend(numpy_data[female1: female1 + female2])
# train.extend(numpy_data[len(numpy_data) - male1 - male2: len(numpy_data) - male1])
# train.extend(numpy_data[female1 + female2: len(numpy_data) - male1 - male2])
# numpy_data = np.asarray(train)
# print(np.sum(numpy_data[:, 0])/len(numpy_data))


#local_aware dataset, party 1: domain 1
#party 2: party1: domain 2

# split_ratio1 = 0.5
# split_ratio2 = 0.5
# party1 = int(batch_size*split_ratio1)
# party2 = int(batch_size*split_ratio2)
# train.extend(numpy_data[0:party1])
# train.extend(numpy_data[len(numpy_data) - party2: len(numpy_data)])
# train.extend(numpy_data[party1: len(numpy_data) - party2])
# numpy_data = np.asarray(train)


index = batch_size
train_data = numpy_data[:, 1:-1][0:index]
train_label = numpy_data[:, -1][0:index]

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
    fair_loss = torch.mean(torch.mul(fair_loss, fair_loss))
    return pred_loss + 4.0*fair_loss

def loss_with_local_fair(output, target, Theta_X, Sen, k):
    pred_loss = torch.mean((-target * torch.log(output)- (1 - target) * torch.log(1 - output)))
    local_length = len(Sen) // k
    fair_loss = 0
    for i in range(k):
        Sen_bar_temp = torch.sum(Sen[i*local_length: (i + 1)*local_length])/local_length
        Sen_temp = Sen[i*local_length: (i + 1)*local_length]
        Theta_X_temp = Theta_X[i*local_length: (i + 1)*local_length]
        # if i == 0:
        #     Sen_temp = Sen[0:5000]
        #     Sen_bar_temp = torch.sum(Sen[0: 5000]) / 7500
        #     Theta_X_temp = Theta_X[0:5000]
        # else:
        #     Sen_temp = Sen[5000:25000]
        #     Sen_bar_temp = torch.sum(Sen[5000: 25000]) / 17500
        #     Theta_X_temp = Theta_X[5000:25000]
        loss_temp = torch.mul(Sen_temp - Sen_bar_temp, Theta_X_temp)
        loss_temp = torch.mean(torch.mul(loss_temp, loss_temp))
        fair_loss = fair_loss + loss_temp
    print(pred_loss)
    print(fair_loss)
    return pred_loss + 1.0*fair_loss



def agnostic_loss_no_fair(output, target, weight):

    pred_loss = (-target * torch.log(output)- (1 - target) * torch.log(1 - output))
    pred_loss = torch.mean(torch.mul(pred_loss, weight))
    return pred_loss


def loss_with_agnostic_local_fair(output, target, Theta_X, weight, Sen, Sen_bar, k):
    pred_loss = (-target * torch.log(output)- (1 - target) * torch.log(1 - output))
    pred_loss = torch.mean(torch.mul(pred_loss, weight))

    fair_loss = 0
    local_length = len(Sen) // k
    for i in range(k):
        weight_temp = weight[i*local_length: (i+1)*local_length]
        Sen_temp = Sen[i*local_length: (i+1)*local_length]
        Theta_X_temp = Theta_X[i*local_length: (i+1)*local_length]
        temp_loss = torch.mul(torch.mul(weight_temp, Sen_temp) - Sen_bar[i], Theta_X_temp)
        fair_loss = fair_loss + torch.mean(torch.mul(temp_loss, temp_loss))

    return pred_loss + 2.0*fair_loss

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


def compute_model_parameter_fairness(weight, Sen, ratio, option, k):
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
            elif option == "loss_with_local_fair":
                loss = loss_with_local_fair(out, label, Theta_X, Sen, k)
            elif option == "loss_with_agnostic_local_fair":
                loss = loss_with_agnostic_local_fair(out, label, Theta_X, weight, Sen, ratio, k)

            print(loss)

            pred = torch.where(out > 0.5, torch.tensor(1.0), torch.tensor(0.0))

            running_acc += torch.sum((pred==label))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Sen_train = numpy_data[:, 0][0:index].reshape(-1, 1)
        risk_difference = np.random.rand(k)
        for j in range(k):
            local_length = len(pred) // k
            Sen_train_j = Sen_train[j*local_length : (j + 1)*local_length]
            pred_j = pred[j*local_length : (j + 1)*local_length]
            label_j = label[j*local_length : (j + 1)*local_length]
            # if j == 0:
            #     Sen_train_j = Sen_train[0: 5000]
            #     pred_j = pred[0: 5000]
            # else:
            #     Sen_train_j = Sen_train[5000: 25000]
            #     pred_j = pred[5000: 25000]
            risk_difference[j] = risk_difference_calculation(Sen_train_j, pred_j)
            accuracy = torch.sum((pred_j==label_j))
            accuracy = accuracy.numpy()
            print(f'Party {i + 1} accuracy {accuracy / local_length:.6f}')
        running_acc = running_acc.numpy()
        global_risk_difference = risk_difference_calculation(Sen_train, pred)
        for i in range(k):
            print(f'Party {i + 1} risk difference {risk_difference[i]:.6f}')
        print(f'global training risk dfference: {global_risk_difference:.6f}')
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
            label = label.view(len(label), -1)
            running_acc += torch.sum((pred == label))

        Sen_test = numpy_data[:, 0][index:].reshape(-1, 1)
        for i in range(k):
            length = len(Sen_test) // k
            # if i == 0:
            #     Sen_train_j = Sen_test[0: 10000]
            #     pred_j = pred[0: 10000]
            #     label_j = label[0:10000]
            # else:
            #     Sen_train_j = Sen_test[10000:20000]
            #     pred_j = pred[10000:20000]
            #     label_j = label[10000:20000]

            Sen_train_j = Sen_test[i*length: (i+1)*length]
            pred_j = pred[i * length: (i + 1) * length]
            label_j = label[i * length: (i + 1) * length]
            risk_difference = risk_difference_calculation(Sen_train_j, pred_j)
            accuracy = torch.sum((pred_j == label_j)).numpy()
            print(f'Finish party {i + 1} test risk difference {risk_difference:.6f}')
            print(f'Party {i + 1} accuracy {accuracy / len(pred_j):.6f}')
        running_acc = running_acc.numpy()
        print(f'Finish {epoch + 1} testing epoch, Acc: {running_acc / len(test_label):.6f}')
        risk_difference = risk_difference_calculation(Sen_test[0:20000], pred[0:20000])
        print(f'Finish {epoch + 1} risk difference {risk_difference:.6f}')


    return theta.detach().numpy().transpose(), bias.detach().numpy()

Iteration = 20
sigma = 2
B = 5
Xtr = train_data
Ytr = train_label
ref_idx = np.random.randint(len(test_data), size = 30)
Xref = test_data[ref_idx, :]
Sen = numpy_data[:, 0][0:index].reshape(-1, 1)
Sen_torch = torch.from_numpy(Sen).float()

weight = torch.ones(batch_size, 1)
ratio = 0.1
options = ["no_fair", "with_fair", "agnostic_fair", "agnostic_loss_no_fair", "loss_with_local_fair", "loss_with_agnostic_local_fair"]
ratio_fair = np.sum(Sen) / len(Sen)
print(ratio_fair)

tau = 1000
k = 2
ratio = np.random.rand(k)
for i in range(k):
    local_length = len(Sen) // k
    ratio[i] = np.sum(Sen[i*local_length: (i+1)*local_length]) / local_length

for i in range(Iteration):
    theta, bias = compute_model_parameter_fairness(weight, Sen_torch, ratio_fair, options[1], k)

    #weight, ratio = compute_alpha_no_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B)
    #weight, ratio = compute_alpha_with_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B, tau)
    #print(weight[0:10])
    #weight = torch.from_numpy(weight).float()
    #ratio = torch.from_numpy(ratio).float()


torch.save(model.state_dict(), './logstic.pth')
#

