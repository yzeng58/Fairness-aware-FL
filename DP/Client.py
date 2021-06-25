# Client for Demographic parity

import torch, random, sys

sys.path.insert(0, '..')

import numpy as np

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import *
from fairBatch import *
import torch.nn.functional as F
from torch.autograd import Variable


################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# my kernel died for some reason
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

class Client(object):
    def __init__(self, dataset, idxs, batch_size, option, seed = 0, prn = True, lbd = None, penalty = 500, Z = 2):
        self.seed = seed 
        self.dataset = dataset
        self.idxs = idxs
        self.option = option
        self.prn = prn
        self.Z = Z
        self.trainloader, self.validloader = self.train_val(dataset, list(idxs), batch_size, lbd)
        self.penalty = penalty

    def train_val(self, dataset, idxs, batch_size, lbd):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)
        
        # split indexes for train, validation (90, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):len(idxs)]
        
        if self.option == "FairBatch": 
            # FairBatch(self, train_dataset, lbd, client_idx, batch_size, replacement = False, seed = 0)
            sampler = FairBatch(DatasetSplit(dataset, idxs_train), lbd, idxs,
                                 batch_size = batch_size, replacement = False, seed = self.seed, Z = self.Z)
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train), sampler = sampler,
                                     batch_size=batch_size, num_workers = 0)
                        
        else:
            trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                     batch_size=batch_size, shuffle=True)

        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def standard_update(self, model, global_round, learning_rate, local_epochs, optimizer): 
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

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
                loss, _, _ = loss_func(self.option, logits, labels, probas, sensitive, self.penalty)
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

    def threshold_adjusting(self, models, learning_rate, local_epochs, optimizer, update_step): 
        z = list(update_step.keys())
        # Set mode to train model
        models[z[0]].train()
        models[z[1]].train()

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = [torch.optim.SGD(models[z[0]].parameters(), lr=0.005) ,
                        torch.optim.SGD(models[z[1]].parameters(), lr=0.005)] 
        elif optimizer == 'adam':
            optimizer = [torch.optim.Adam(models[z[0]].parameters(), lr=0.005, weight_decay=1e-4),
                        torch.optim.Adam(models[z[1]].parameters(), lr=0.005, weight_decay=1e-4)]
        bias_grad = 0
        for epoch in range(local_epochs):
            # only update the bias
            param0 = next(models[z[0]].parameters())
            param0.requires_grad = False

            param1 = next(models[z[1]].parameters())
            param1.requires_grad = False

            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs
                
                group_idx, loss, bias_grad, w = [], [0,0], [0,0], []
                for i in [0,1]:
                    w.append(copy.deepcopy(models[z[i]]).state_dict())
                    group_idx.append(torch.where(sensitive == z[i]))

                    probas, logits = models[z[i]](features[group_idx[i]])
                    loss[i], _, _ = loss_func(self.option, logits, labels[group_idx[i]], probas, sensitive[group_idx[i]])
                    optimizer[i].zero_grad()
                    # compute the gradients
                    loss[i].backward()
                    # get the gradient of the bias
                    bias_grad[i] = models[z[0]].linear.bias.grad[1]

                update_idx = np.argmin(bias_grad)
                w[update_idx]['linear.bias'][1] += learning_rate * update_step[z[update_idx]] / (epoch + 1) ** .5
                w[update_idx]['linear.bias'][0] -= learning_rate * update_step[z[update_idx]] / (epoch + 1) ** .5
                models[z[update_idx]].load_state_dict(w[update_idx])
        # weight, loss
        # print(models[z[0]].state_dict(), models[z[1]].state_dict())
        return models[z[0]].state_dict(), models[z[1]].state_dict()

    def bc_update(self, model, mu, global_round, learning_rate, local_epochs, optimizer):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE).type(torch.LongTensor)
                
                v = torch.randn(len(labels)).type(torch.DoubleTensor)
                # labels_i == 1
                y1_idx = torch.where(labels == 1)[0]
                exp = np.exp(torch.index_select(mu, 0, sensitive[y1_idx].type(torch.LongTensor)))
                v[y1_idx] = exp / (1 + exp)

                # labels_i == 0
                y0_idx = torch.where(labels == 0)[0]
                exp = np.exp(torch.index_select(mu, 0, sensitive[y0_idx].type(torch.LongTensor)))
                v[y0_idx] = 1 / (1 + exp)

                _, logits = model(features)
                loss = weighted_loss(logits, labels, v)
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

    def bc_compute(self, model, mu):
        model.eval()

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        idxs_train = self.idxs[:int(0.9*len(self.idxs))]
        trainloader = DatasetSplit(self.dataset, idxs_train)      
        x, y, z = torch.tensor(trainloader.x), torch.tensor(trainloader.y), torch.tensor(trainloader.sen)
        x = x.to(DEVICE)

        probas, _ = model(x)
        probas = probas.T[1] / torch.sum(probas, dim = 1)

        v = torch.ones(len(y)).type(torch.DoubleTensor)

        n, nz, yz, yhat = v.sum().item(), torch.zeros(self.Z).type(torch.DoubleTensor), torch.zeros(self.Z).type(torch.DoubleTensor), (probas * v).sum().item()
        z_idx = []
        for z_ in range(self.Z):
            z_idx.append(torch.where(z_ == z)[0])
            yz[z_] = (probas[z_idx[z_]] * v[z_idx[z_]]).sum().item()
            nz[z_] = v[z_idx[z_]].sum().item()

        return n, nz, yz, yhat

    def postbc_compute(self, model, mu):
        model.eval()

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        trainloader = DatasetSplit(self.dataset, self.idxs)      
        x, y, z = torch.tensor(trainloader.x), torch.tensor(trainloader.y), torch.tensor(trainloader.sen)
        x = x.to(DEVICE)

        probas, _ = model(x)
        probas = probas.T[1] / torch.sum(probas, dim = 1)

        n, nz, yz = len(y), torch.zeros(self.Z).type(torch.DoubleTensor), torch.zeros(self.Z).type(torch.DoubleTensor)
        for y_ in [0,1]:
            for z_ in range(self.Z):
                yz_idx = torch.where((z_ == z) & (y == y_))[0]
                if y_ == 1:
                    probas[yz_idx] = probas[yz_idx]/(probas[yz_idx] + (1-probas[yz_idx]) * np.exp(-mu[z_]))
                yz[z_] += probas[yz_idx].sum().item()
                nz[z_] += len(yz_idx)
        yhat = probas.sum().item()
        return n, nz, yz, yhat

    def bc3_update(self, model, mu, global_round, learning_rate, local_epochs, optimizer):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE).type(torch.LongTensor)
                
                v = torch.randn(len(labels)).type(torch.DoubleTensor)
                # labels_i == 1
                y1_idx = torch.where(labels == 1)[0]
                exp = np.exp(torch.index_select(mu, 0, sensitive[y1_idx].type(torch.LongTensor)))
                v[y1_idx] = exp

                # labels_i == 0
                y0_idx = torch.where(labels == 0)[0]
                exp = np.exp(torch.index_select(mu, 0, sensitive[y0_idx].type(torch.LongTensor)))
                v[y0_idx] = 1 

                _, logits = model(features)
                loss = weighted_loss(logits, labels, v)
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

    def bc1_compute(self, model, mu):
        model.eval()

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        trainloader = DatasetSplit(self.dataset, self.idxs)      
        x, y, z = torch.tensor(trainloader.x), torch.tensor(trainloader.y), torch.tensor(trainloader.sen)
        x = x.to(DEVICE)

        probas, _ = model(x)
        _, pred_labels = torch.max(probas, 1)
        prob = torch.zeros(self.Z)
        for z_ in range(self.Z): 
            prob[z_] = torch.sum((pred_labels == 1) & (z == z_)).item() / torch.sum(z == z_).item()

        v = torch.ones(len(y)).type(torch.DoubleTensor)

        n, nz, yz = v.sum().item(), torch.zeros(self.Z).type(torch.DoubleTensor), torch.zeros(self.Z).type(torch.DoubleTensor)
        z_idx, yhat = [], 0
        for z_ in range(self.Z): 
            z_idx.append(torch.where(z == z_)[0])
            yz[z_] = (prob[z_] * v[z_idx[z_]]).sum().item()
            nz[z_] = v[z_idx[z_]].sum().item()
            yhat += yz[z_]
        return n, nz, yz, yhat

    def mean_sensitive_stat(self): 
        trainloader = DatasetSplit(self.dataset, self.idxs)      
        z = torch.tensor(trainloader.sen)
        return z.sum().item(), len(z)

    def zafar_update(self, model, global_round, learning_rate, local_epochs, optimizer, mean_z, left = True):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

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
                loss = zafar_loss(logits, labels, probas, sensitive, self.penalty, mean_z, left)
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

    def al_update(self, model, adv_model, global_round, learning_rate, local_epochs, optimizer, alpha): 
        # Set mode to train model
        model.train()
        epoch_loss = []
        
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            adv_optimizer = torch.optim.SGD(adv_model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5     

            optimizer1 = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimizer2 = torch.optim.SGD(model.parameters(), lr=learning_rate * alpha)
        elif optimizer == 'adam':
            adv_optimizer = torch.optim.Adam(adv_model.parameters(), lr=learning_rate, weight_decay=1e-4)
            optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate * alpha, weight_decay=1e-4)

        for i in range(local_epochs):
            learning_rate = learning_rate * (1. / (1. + 1e-4 * (global_round*local_epochs + i)))
            alpha = alpha * (1. / (1. + 1e-4 * (global_round*local_epochs + i)))

            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                
                _, logits = model(features)
                # _, pred_labels = torch.max(probas, 1)
                # pred_label_matrix = torch.tensor([1-pred_labels, pred_labels]).T

                # _, adv_logits = adv_model(pred_labels.reshape(labels.shape[0],1))  
                _, adv_logits = adv_model(logits)       
                loss, adv_loss = al_loss(logits, labels, adv_logits, sensitive)
                neg_adv_loss = -adv_loss

                # classification model
                # w = w - lr * partial predictor_loss / partial w
                optimizer1.zero_grad()
                loss.backward(retain_graph = True)
                g1 = torch.cat((model.linear.weight.grad.T, model.linear.bias.grad.reshape(1,2)), dim = 0).T

                # classification model
                # w = w - lr * (-alpha * partial adversary_loss / partial w)
                optimizer2.zero_grad()
                neg_adv_loss.backward(retain_graph = True)
                g2 = -torch.cat((model.linear.weight.grad.T, model.linear.bias.grad.reshape(1,2)), dim = 0).T

                # adversarial model
                adv_optimizer.zero_grad()
                adv_loss.backward()

                # update
                # optimizer1.step()
                # optimizer2.step()
                adv_optimizer.step()

                # classification model
                # w = w - lr * (- proj partial adversary_loss / partial w, partial predictor_loss / partial w)
                w = model.state_dict()
                proj = torch.mul(g1[0], g2[0]) / torch.sqrt(torch.mul(g2[0], g2[0]))

                w['linear.weight'][0] = w['linear.weight'][0] + learning_rate * proj[:-1] - learning_rate * g1[0][:-1] + alpha * learning_rate * g2[0][:-1]
                w['linear.weight'][1] = w['linear.weight'][1] - learning_rate * proj[:-1] - learning_rate * g1[1][:-1] + alpha * learning_rate * g2[1][:-1]
                w['linear.bias'][0] = w['linear.bias'][0] + learning_rate * proj[-1] - learning_rate * g1[0][-1] + alpha * learning_rate * g2[0][-1]
                w['linear.bias'][1] = w['linear.bias'][0] - learning_rate * proj[-1] - learning_rate * g1[1][-1] + alpha * learning_rate * g2[1][-1]

                # update
                model.load_state_dict(w)

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), adv_model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def al_inference(self, model, adv_model):
        model.eval()
        total, correct, acc_loss, adv_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0
        
        for _, (features, labels, sensitive) in enumerate(self.validloader):
            features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
            sensitive = sensitive.to(DEVICE).type(torch.LongTensor)
            
            # Inference
            outputs, logits = model(features)
            _, adv_logits = adv_model(logits)

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
            
            batch_acc_loss, batch_adv_loss = al_loss(logits, labels, adv_logits, sensitive)
            acc_loss, adv_loss = (acc_loss + batch_acc_loss.item(), 
                                         adv_loss + batch_adv_loss.item())
        accuracy = correct/total
        return accuracy, n_yz, acc_loss / num_batch, adv_loss / num_batch

    def ftrain_update(self, generator, discriminator, global_round, lr_g, lr_d, local_epochs, ratio_gd, lambda_d, init_epochs = 100):
        # Set mode to train model
        generator.train()
        epoch_loss = []
        
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr_g)
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = lr_d)

        for i in range(local_epochs):
            batch_loss = []
            epoch = global_round * local_epochs + i

            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE).type(torch.LongTensor)

                # for the first 100 epochs, train both generator and discriminator
                # after the first 100 epochs, ratio of updating generator and discriminator (1:ratio_gd training)
                if epoch % ratio_gd == 0 or epoch < init_epochs:
                    # forward generator
                    optimizer_G.zero_grad()
                
                _, logits_g = generator(features)

                # train fairness discriminator
                optimizer_D.zero_grad()
                loss_d = F.cross_entropy(discriminator(logits_g.detach())[1], sensitive)
                loss_d.backward()
                optimizer_D.step()

                loss_g = F.cross_entropy(logits_g, labels)
                # update generator
                if epoch < init_epochs:
                    loss_g.backward()
                    optimizer_G.step()

                elif epoch % ratio_gd == 0:
                    _, logits_d = discriminator(logits_g)
                    loss_d = F.cross_entropy(logits_d, sensitive)
                    
                    loss = (1-lambda_d) * loss_g - lambda_d * loss_d
                    loss.backward()
                    optimizer_G.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss_g.item()))
                batch_loss.append(loss_g.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return generator.state_dict(), discriminator.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def fb_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                v = torch.randn(len(labels)).type(torch.DoubleTensor)
                
                group_idx = {}
                
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                    nc += v[group_idx[(y,z)]].sum().item()

                # print(logits)
                loss = weighted_loss(logits, labels, v)
                # if global_round == 1: print(loss)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def fb2_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                v = torch.ones(len(labels)).type(torch.DoubleTensor)
                
                group_idx = {}
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                    nc += v[group_idx[(y,z)]].sum().item()

                loss = weighted_loss(logits, labels, v, False)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def get_n_yz(self):
        trainloader = DatasetSplit(self.dataset, self.idxs)      
        x, y, z = torch.tensor(trainloader.x), torch.tensor(trainloader.y), torch.tensor(trainloader.sen)
        x = x.to(DEVICE)

        n_yz = {}
        for y_ in [0,1]:
            for z_ in range(self.Z):
                n_yz[(y_,z_)] = 0
        for y_, z_ in n_yz:
            n_yz[(y_,z_)] = torch.sum((y == y_) & (z == z_)).item()

        return n_yz

    def inference(self, model, train = False):
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
        n_yz, loss_yz = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
        
        dataset = self.validloader if not train else self.trainloader
        for _, (features, labels, sensitive) in enumerate(dataset):
            features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
            sensitive = sensitive.to(DEVICE).type(torch.LongTensor)
            
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
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()     
                
                if self.option == "FairBatch":
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx[yz]], 
                                                    labels[group_boolean_idx[yz]], 
                                         outputs[group_boolean_idx[yz]], sensitive[group_boolean_idx[yz]], 
                                         self.penalty)
                    loss_yz[yz] += loss_yz_
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits, 
                                                        labels, outputs, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if self.option in ["FairBatch", "FB-Variant1"]:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None

    def postbc_inference(self, model, mu):
        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
        
        # dataset = self.validloader if option != "FairBatch" else self.dataset
        for _, (features, labels, sensitive) in enumerate(self.validloader):
            features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
            sensitive = sensitive.to(DEVICE)
            
            # Inference
            outputs, logits = model(features)
            probas = torch.ones(outputs.shape).T
            probas[0] = outputs.T[0] / torch.sum(outputs, dim = 1)
            probas[1] = outputs.T[1] / torch.sum(outputs, dim = 1)

            # Prediction
            for y in [0,1]:
                for z_ in range(self.Z):
                    yz_idx = torch.where((z_ == sensitive) & (labels == y))[0]
                    if y == 1:
                        probas[1][yz_idx] = probas[1][yz_idx]/(probas[1][yz_idx] + (1-probas[1][yz_idx]) * np.exp(-mu[z_]))
                        probas[0][yz_idx] = probas[0][yz_idx]/(probas[0][yz_idx] + (1-probas[0][yz_idx]) * np.exp(mu[z_]))  

            _, pred_labels = torch.max(probas.T, 1)
            pred_labels = pred_labels.view(-1)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1
            
            group_boolean_idx = {}
            
            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()     
            
            logits = logit_compute(probas.T)
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits, 
                                                        labels, probas.T, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                            acc_loss + batch_acc_loss.item(), 
                                            fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None