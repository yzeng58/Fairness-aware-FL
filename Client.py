import torch, copy, time, random, warnings

import numpy as np

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import *
from fairBatch import *
import torch.nn.functional as F

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# my kernel died for some reason
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

class ClientUpdate(object):
    def __init__(self, dataset, idxs, batch_size, option, penalty = 0, lbd = None, seed = 0, mean_sensitive = None, prn = True):
        self.trainloader, self.validloader = self.train_val(dataset, list(idxs), batch_size, option, lbd, seed)
        self.dataset = dataset
        self.option = option
        self.penalty = penalty
        self.mean_sensitive = mean_sensitive
        self.prn = prn
            
    def train_val(self, dataset, idxs, batch_size, option, lbd, seed):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(seed)
        
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
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs
                
                probas, logits = model(features)
                loss, _, _ = loss_func(self.option,
                    logits, labels, probas, sensitive, self.mean_sensitive, self.penalty)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def threshold_adjust(self, model, global_round, learning_rate, local_epochs, optimizer):
        # Set mode to train model
        model.train()
        bias_grad = 0

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                         weight_decay=1e-4)

        for _ in range(local_epochs):
            # only update the bias
            param = next(model.parameters())
            param.requires_grad = False

            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs
                
                probas, logits = model(features)
                loss, _, _ = loss_func(self.option,
                    logits, labels, probas, sensitive, self.mean_sensitive, self.penalty)
                    
                optimizer.zero_grad()
                # compute the gradients
                loss.backward()
                # get the gradient of the bias
                bias_grad += model.linear.bias.grad
                # paarameter update
                optimizer.step()

        # weight, loss
        return model.state_dict(), bias_grad

    def standard_inference(self, model, option):
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
        
        # dataset = self.validloader if option != "FairBatch" else self.dataset
        for _, (features, labels, sensitive) in enumerate(self.validloader):
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
                    loss_yz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx[yz]], 
                                                    labels[group_boolean_idx[yz]], 
                                         outputs[group_boolean_idx[yz]], sensitive[group_boolean_idx[yz]], 
                                         self.mean_sensitive, self.penalty)
                    loss_yz[yz] += loss_yz_
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits, 
                                                        labels, outputs, sensitive, self.mean_sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if option == "FairBatch":
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None

    def FairBatch_inference(self):
        pas