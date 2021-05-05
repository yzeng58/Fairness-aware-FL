import torch, copy, time, random, warnings

import numpy as np

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from Client import *
from utils import *
from fairBatch import *

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#########################################################

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False, 
                train_prn = False, metric = "Risk Difference", 
                batch_size = 128, print_every = 1, fraction_clients = 1):
        """
        Server execution.

        Parameters
        ----------
        model: torch.nn.Module object.

        dataset_info: a list of three objects.
            - train_dataset: Dataset object.
            - test_dataset: Dataset object.
            - clients_idx: a list of lists, with each sublist contains the indexs of the training samples in one client.
                    the length of the list is the number of clients.

        seed: random seed.

        num_workers: number of workers.

        ret: boolean value. If true, return the accuracy and fairness measure and print nothing; else print the log and return None.

        train_prn: boolean value. If true, print the batch loss in local epochs.

        metric: three options, "Risk Difference", "pRule", "Demographic disparity".

        batch_size: a positive integer.

        print_every: a positive integer. eg. print_every = 1 -> print the information of that global round every 1 round.

        fraction_clients: float from 0 to 1. The fraction of clients chose to update the weights in each round.
        """

        self.model = model
        self.seed = seed
        self.num_workers = num_workers

        self.ret = ret
        self.prn = not ret
        self.train_prn = False if ret else train_prn

        self.metric = metric
        if metric == "Risk Difference":
            self.disparity = riskDifference
        elif metric == "pRule":
            self.disparity = pRule
        elif metric == "Demographic disparity":
            self.disparity = DPDisparity
        else:
            warnings.warn("Warning message: metric " + metric + " is not supported! Use the default metric risk difference. ")
            self.disparity = riskDifference
            self.metric = "Risk Difference"

        self.batch_size = batch_size
        self.print_every = print_every
        self.fraction_clients = fraction_clients

        self.train_dataset, self.test_dataset, self.clients_idx = dataset_info
        self.num_clients = len(self.clients_idx)

    def Unconstrained(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, 
                    optimizer = "adam", epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(local_weights, self.clients_idx, idxs_users)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if epsilon: 
                if self.disparity(n_yz) < epsilon: break

        # Test inference after completion of training
        test_acc, n_yz= self.test_inference()
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    # post-processing approach
    def ThresholdAdjust(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.7, 
                        epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        start_time = time.time()
        models = [copy.deepcopy(self.model), copy.deepcopy(self.model)]
        bias_grad, w = [0,0], [0,0]
        clients_idx_sen = {0:[], 1:[]}

        for c in range(self.num_clients):
            for sen in [0,1]:
                clients_idx_sen[sen].append(self.clients_idx[c][self.train_dataset.sen[self.clients_idx[c]] == sen])
        
        _, n_yz = self.test_inference()
        for round_ in tqdm(range(num_rounds)):
            rd = riskDifference(n_yz, False)
            if self.disparity(n_yz) <= epsilon: break
            update_step = {0:1, 1:-1} if rd > 0 else {0:-1, 1:1}

            local_weights = [[],[]]

            for c in range(self.num_clients):
                for sen in [0,1]:
                    local_model = Client(dataset=self.train_dataset,
                                                idxs=clients_idx_sen[sen][c], batch_size = self.batch_size, 
                                            option = "threshold adjusting",  
                                            seed = self.seed, prn = self.train_prn)
                    w[sen], bias_grad[sen] = local_model.threshold_adjusting(
                                    model=copy.deepcopy(models[sen]), global_round=round_, 
                                        learning_rate = 0.005, local_epochs = local_epochs, 
                                        optimizer = 'adam')
                
                update_sen = 0 if bias_grad[0][1] < bias_grad[1][1] else 1
                w[update_sen]['linear.bias'][1] += learning_rate * update_step[update_sen]
                w[update_sen]['linear.bias'][0] -= learning_rate * update_step[update_sen]
                local_weights[update_sen].append(copy.deepcopy(w[update_sen]))
                local_weights[1-update_sen].append(copy.deepcopy(models[1-update_sen].state_dict()))
            # update global weights
            if len(local_weights[0]): models[0].load_state_dict(average_weights(local_weights[0], clients_idx_sen[0], range(self.num_clients)))
            if len(local_weights[1]): models[1].load_state_dict(average_weights(local_weights[1], clients_idx_sen[1], range(self.num_clients)))

            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}

            train_loss, train_acc = [], []
            for c in range(self.num_clients):
                for sen in [0,1]:
                    local_model = Client(dataset=self.train_dataset,
                                                idxs=clients_idx_sen[sen][c], batch_size = self.batch_size, 
                                            option = "threshold adjusting",  
                                            seed = self.seed, prn = self.train_prn)
                    acc, loss, n_yz_c, _, _, _ = local_model.inference(model = models[sen])
                    train_loss.append(loss)
                    train_acc.append(acc)

                    for yz in n_yz:
                        n_yz[yz] += n_yz_c[yz]            

            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} threshold adjusting global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*np.array(train_acc).mean(), self.metric, self.disparity(n_yz)))
                
            if epsilon: 
                if self.disparity(n_yz) < epsilon: break
  
        # Test inference after completion of training
        n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        idx_0 = np.where(self.test_dataset.sen == 0)[0].tolist()
        idx_1 = np.where(self.test_dataset.sen == 1)[0].tolist()
        test_acc_0, n_yz_0  = self.test_inference(models[0], DatasetSplit(self.test_dataset, idx_0))
        test_acc_1, n_yz_1 = self.test_inference(models[1], DatasetSplit(self.test_dataset, idx_1))
        
        for yz in n_yz:
            n_yz[yz] = n_yz_0[yz] + n_yz_1[yz]

        test_acc = (test_acc_0 * len(idx_0) + test_acc_1 * len(idx_1)) / (len(idx_0) + len(idx_1))
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    
    def Zafar(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.001, penalty = 500, optimizer = 'adam', epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "Zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(local_weights, self.clients_idx, idxs_users)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx],
                            batch_size = self.batch_size, option = "Zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))
            
            if epsilon: 
                if self.disparity(n_yz) < epsilon: break

        # Test inference after completion of training
        test_acc, n_yz= self.test_inference()
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    def FairBatch(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 1, adaptive_alpha = True):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        # the number of samples whose label is y and sensitive attribute is z
        m_yz = {(0,0): ((self.train_dataset.y == 0) & (self.train_dataset.sen == 0)).sum(),
            (1,0): ((self.train_dataset.y == 1) & (self.train_dataset.sen == 0)).sum(),
            (0,1): ((self.train_dataset.y == 0) & (self.train_dataset.sen == 1)).sum(),
            (1,1): ((self.train_dataset.y == 1) & (self.train_dataset.sen == 1)).sum()}

        lbd = {
            (0,0): m_yz[(0,0)]/(m_yz[(0,1)] + m_yz[(0,0)]), 
            (0,1): m_yz[(0,1)]/(m_yz[(0,1)] + m_yz[(0,0)]),
            (1,0): m_yz[(1,0)]/(m_yz[(1,1)] + m_yz[(1,0)]),
            (1,1): m_yz[(1,1)]/(m_yz[(1,1)] + m_yz[(1,0)]),
        }
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                        option = "FairBatch", lbd = lbd, 
                                        seed = self.seed, prn = self.train_prn)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(local_weights, self.clients_idx, idxs_users)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            loss_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FairBatch", 
                                            lbd = lbd, seed = self.seed, prn = self.train_prn)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    loss_yz[yz] += loss_yz_c[yz]
                    
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))
                
            # update the lambda according to the paper -> see Section A.1 of FairBatch
            # works well! The real batch size would be slightly different from the setting
            loss_yz[(0,0)] = loss_yz[(0,0)]/(m_yz[(0,0)] + m_yz[(1,0)])
            loss_yz[(1,0)] = loss_yz[(1,0)]/(m_yz[(0,0)] + m_yz[(1,0)])
            loss_yz[(0,1)] = loss_yz[(0,1)]/(m_yz[(0,1)] + m_yz[(1,1)])
            loss_yz[(1,1)] = loss_yz[(1,1)]/(m_yz[(0,1)] + m_yz[(1,1)])
            
            y0_diff = abs(loss_yz[(0,0)] - loss_yz[(0,1)])
            y1_diff = abs(loss_yz[(1,0)] - loss_yz[(1,1)])
            if y1_diff < y0_diff:
                lbd[(0,0)] += alpha * (2*int((loss_yz[(0,1)] - loss_yz[(0,0)]) > 0)-1)
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(0,1)] = 1 - lbd[(0,0)]
            else:
                lbd[(1,0)] -= alpha * (2*int((loss_yz[(1,1)] - loss_yz[(1,0)]) > 0)-1)
                lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                lbd[(1,1)] = 1 - lbd[(1,0)]

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if adaptive_alpha: alpha = DPDisparity(n_yz)

        # Test inference after completion of training
        test_acc, n_yz = self.test_inference(self.model, self.test_dataset)
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd


    def BiasCorrecting(self):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def test_inference(self, model = None, test_dataset = None):
        """ 
        Returns the test accuracy and fairness level.
        """
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if model == None: model = self.model
        if test_dataset == None: test_dataset = self.test_dataset

        model.eval()
        total, correct = 0.0, 0.0
        n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
        
        testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False)

        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels =  labels.to(DEVICE).type(torch.LongTensor)
            # Inference
            outputs, _ = model(features)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            
            for yz in n_yz:
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()  

        accuracy = correct/total

        return accuracy, n_yz