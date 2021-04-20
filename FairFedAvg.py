import torch, copy, time, random

import pandas as pd
import numpy as np

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from load_adult import *
from utils import *
from fairBatch import *
import torch.nn.functional as F
from torch.autograd import Variable

from functools import partial

class ClientUpdate(object):
    def __init__(self, dataset, idxs, batch_size, option, penalty = 0, lbd = None, seed = 0):
        self.trainloader, self.validloader = self.train_val(dataset, list(idxs), batch_size, option, lbd, seed)
        self.dataset = dataset
        self.option = option
        self.penalty = penalty
            
    def train_val(self, dataset, idxs, batch_size, option, lbd, seed):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        
        # split indexes for train, validation (90, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):len(idxs)]
        
        if option == "FairBatch": 
            # FairBatch(self, train_dataset, lbd, client_idx, batch_size, replacement = False, seed = 0)
            sampler = FairBatch(DatasetSplit(dataset, idxs_train), lbd, idxs,
                                 batch_size = batch_size, replacement = False, seed = seed)
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train), sampler = sampler,
                                     batch_size=batch_size, num_workers = 0)
                        
        else:
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=batch_size, shuffle=True)

        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=int(len(idxs_val)/10), shuffle=False)
        return trainloader, validloader

    def update_weights(self, model, global_round, learning_rate, local_epochs, optimizer):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                         weight_decay=1e-4)

        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs

                log_probs, logits = model(features)
                loss, _, _ = loss_func(self.option,
                    logits, labels, logits, sensitive, mean_sensitive, self.penalty)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, option):
        """ 
        Returns the inference accuracy, 
                                loss, 
                                N(sensitive group, pos), 
                                N(non-sensitive group, pos), 
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        loss_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        
        dataset = self.validloader if option != "FairBatch" else self.dataset
        for batch_idx, (features, labels, sensitive) in enumerate(self.validloader):
            features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
            sensitive = sensitive.to(DEVICE)
            
            # Inference
            outputs, logits = model(features)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1
            
            group_boolean_idx = {}
            
            
            for yz in n_yz:
                group_boolean_idx[yz] = (pred_labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum(group_boolean_idx[yz]).item()            
                
                if self.option == "FairBatch":
                # the objective function have no lagrangian term
                    loss_yz_,_,_ = loss_func("unconstrained", outputs[group_boolean_idx[yz]], 
                                                    labels[group_boolean_idx[yz]], 
                                         logits[group_boolean_idx[yz]], sensitive[group_boolean_idx[yz]], 
                                         mean_sensitive, self.penalty)
                    loss_yz[yz] += loss_yz_
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, outputs, 
                                                        labels, logits, sensitive, mean_sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if option == "FairBatch":
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None

# TODO: need to update the criterion to loss_func
def test_inference(model, test_dataset, batch_size):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
    
    criterion = nn.NLLLoss().to(DEVICE)
    testloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False)

    for batch_idx, (features, labels, sensitive) in enumerate(testloader):
        features = features.to(DEVICE)
        labels =  labels.to(DEVICE).type(torch.LongTensor)
        # Inference
        outputs, logits = model(features)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        bool_correct = torch.eq(pred_labels, labels)
        correct += torch.sum(bool_correct).item()
        total += len(labels)
        
        for yz in n_yz:
            n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()  

    accuracy = correct/total
    # |P(Group1, pos) - P(Group2, pos)| = |N(Group1, pos)/N(Group1) - N(Group2, pos)/N(Group2)|
    return accuracy, loss, RD(n_yz)

def train(model, option = "unconstrained", batch_size = 128, num_clients = 2,
          num_rounds = 5, learning_rate = 0.01, optimizer = 'adam', local_epochs= 5, 
          num_workers = 4, print_every = 1,
         penalty = 1, alpha = 0.005, seed = 123):
    """
    Server execution.
    """
    
    np.random.seed(seed)
    random.seed(seed)
        
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    start_time = time.time()
    weights = model.state_dict()
    
    test_loader = DataLoader(dataset = test_dataset,
                            batch_size = batch_size,
                            num_workers = num_workers)
    
    train_loader = DataLoader(dataset = train_dataset,
                        batch_size = batch_size,
                        num_workers = num_workers)

    def average_weights(w):
        """
        Returns the average of the weights.
        """
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg

    # the number of samples whose label is y and sensitive attribute is z
    m_yz = {(0,0): ((train_dataset.y == 0) & (train_dataset.sen == 0)).sum(),
           (1,0): ((train_dataset.y == 1) & (train_dataset.sen == 0)).sum(),
           (0,1): ((train_dataset.y == 0) & (train_dataset.sen == 1)).sum(),
           (1,1): ((train_dataset.y == 1) & (train_dataset.sen == 1)).sum()}

    lbd = {
        (0,0): m_yz[(0,0)]/train_dataset.y.shape[0], 
        (0,1): m_yz[(0,1)]/train_dataset.y.shape[0],
        (1,0): m_yz[(0,1)]/train_dataset.y.shape[0],
        (1,1): m_yz[(1,1)]/train_dataset.y.shape[0],
    }
    
    
    for round_ in tqdm(range(num_rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {round_+1} |\n')

        model.train()
        m = 2 # the number of clients to be chosen in each round_
        idxs_users = np.random.choice(range(num_clients), m, replace=False)

        for idx in idxs_users:
            local_model = ClientUpdate(dataset=train_dataset,
                                        idxs=clients_idx[idx], batch_size = batch_size, 
                                       option = option, penalty = penalty, lbd = lbd, seed = seed)
            w, loss = local_model.update_weights(
                            model=copy.deepcopy(model), global_round=round_, 
                                learning_rate = learning_rate, local_epochs = local_epochs, 
                                optimizer = optimizer)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        weights = average_weights(local_weights)
        model.load_state_dict(weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all clients at every round
        list_acc, list_loss = [], []
        # the number of samples which are assigned to class y and belong to the sensitive group z
        n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        loss_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        model.eval()
        for c in range(m):
            local_model = ClientUpdate(dataset=train_dataset,
                                        idxs=clients_idx[c], batch_size = batch_size, option = option, 
                                       penalty = penalty, lbd = lbd, seed = seed)
            # validation dataset inference
            acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = model, 
                                                                                      option = option) 
            list_acc.append(acc)
            list_loss.append(loss)
            
            for yz in n_yz:
                n_yz[yz] += n_yz_c[yz]
                
                if option == "FairBatch": loss_yz[yz] += loss_yz_c[yz]
                
            print("Client %d: accuracy loss: %.2f | fairness loss %.2f | RD = %.2f = |%d/%d-%d/%d| " % (
                c, acc_loss, fair_loss, RD(n_yz_c), n_yz_c[(1,1)], n_yz_c[(1,1)] + n_yz_c[(0,1)], 
                n_yz_c[(1,0)], n_yz_c[(1,0)] + n_yz_c[(0,0)]))
            
        if option == "FairBatch": 
            # update the lambda according to the code 
            # inconsistent with the theory
            # work very well! -> even better than the one below
            # y0_diff = abs(loss_yz[(0,0)]/m_yz[(0,0)] - loss_yz[(0,1)]/m_yz[(0,1)])
            # y1_diff = abs(loss_yz[(1,0)]/m_yz[(1,0)] - loss_yz[(1,1)]/m_yz[(1,1)])
            # if y1_diff < y0_diff:
            #     lbd[(0,0)] -= alpha * (2*int((loss_yz[(0,1)]/m_yz[(0,1)] - loss_yz[(0,0)]/m_yz[(0,0)]) > 0)-1)
            #     lbd[(0,0)] = max(0, lbd[(0,0)])
            #     lbd[(0,1)] = 1 - lbd[(0,0)]
            # else:
            #     lbd[(1,0)] -= alpha * (2*int((loss_yz[(1,1)]/m_yz[(1,1)] - loss_yz[(1,0)]/m_yz[(1,0)]) > 0)-1)
            #     lbd[(1,0)] = max(0, lbd[(1,0)])
            #     lbd[(1,1)] = 1 - lbd[(1,0)]

            # update the lambda according to the paper -> see Section A.1 of FairBatch
            # works well! The real batch size would be different from the setting
            loss_yz[(0,0)] = (loss_yz[(0,0)] - 1)/(m_yz[(0,0)] + m_yz[(1,0)])
            loss_yz[(1,0)] = loss_yz[(1,0)]/(m_yz[(0,0)] + m_yz[(1,0)])
            loss_yz[(0,1)] = (loss_yz[(0,1)] - 1)/(m_yz[(0,1)] + m_yz[(1,1)])
            loss_yz[(1,1)] = loss_yz[(1,1)]/(m_yz[(0,1)] + m_yz[(1,1)])

            y0_diff = abs(loss_yz[(0,0)] - loss_yz[(0,1)])
            y1_diff = abs(loss_yz[(1,0)] - loss_yz[(1,1)])
            if y1_diff < y0_diff:
                lbd[(0,0)] -= alpha * (2*int((loss_yz[(0,1)] - loss_yz[(0,0)]) > 0)-1)
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(0,1)] = 1 - lbd[(0,0)]
            else:
                lbd[(1,0)] -= alpha * (2*int((loss_yz[(1,1)] - loss_yz[(1,0)]) > 0)-1)
                lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                lbd[(1,1)] = 1 - lbd[(1,0)]

        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (round_+1) % print_every == 0:
            print(f' \nAvg Training Stats after {round_+1} global rounds:')
            if option != "FairBatch":
                print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation RD: %.4f" % (
                     np.mean(np.array(train_loss)), 
                    100*train_accuracy[-1], RD(n_yz)))
            else:
                print("Training loss: %.2f | Training accuracy: %.2f%% | Training RD: %.4f" % (
                     np.mean(np.array(train_loss)), 
                    100*train_accuracy[-1], RD(n_yz)))

    # Test inference after completion of training
    test_acc, test_loss, rd= test_inference(model, test_dataset, batch_size)

    print(f' \n Results after {num_rounds+1} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Compute RD: risk difference - fairness metric
    # |P(Group1, pos) - P(Group2, pos)| = |N(Group1, pos)/N(Group1) - N(Group2, pos)/N(Group2)|
    print("|---- Test RD: {:.4f}".format(rd))

    print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))