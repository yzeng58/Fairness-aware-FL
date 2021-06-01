# Server for Demographic parity

import torch, copy, time, random, warnings, sys

import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from Client import *

sys.path.insert(0, '..')
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
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
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
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

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
 
    def LocalZafar(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.001, penalty = 500, optimizer = 'adam', epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = copy.deepcopy(self.model).state_dict()
        best_state, lowest_dp = weights, 100
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "local zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)

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
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                            batch_size = self.batch_size, option = "local zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)
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
            
            if self.disparity(n_yz) < lowest_dp and 100*train_accuracy[-1] > 50:
                lowest_dp = self.disparity(n_yz)
                best_state = copy.deepcopy(self.model).state_dict()

            if epsilon: 
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

        self.model.load_state_dict(best_state)
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

    def AdversarialLearning(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, 
                    optimizer = "adam", epsilon = None, alpha = 0.005, sensitive_level = 2, num_classes = 2, adaptive_lr = False):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # adversarial model: learn the sensitive attribute
        adv_model = logReg(num_classes, sensitive_level)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_adv_weights, local_losses = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_client = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "al", seed = self.seed, prn = self.train_prn)

                w, adv_w, loss = local_client.al_update(
                                model=copy.deepcopy(self.model), adv_model = adv_model, global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, alpha = alpha)
                local_weights.append(copy.deepcopy(w))
                local_adv_weights.append(copy.deepcopy(adv_w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(local_weights, self.clients_idx, idxs_users)
            adv_weights = average_weights(local_adv_weights, self.clients_idx, idxs_users)
            self.model.load_state_dict(weights)
            adv_model.load_state_dict(adv_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_client = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn)
                # validation dataset inference
                acc, n_yz_c, acc_loss, adv_loss = local_client.al_inference(self.model, adv_model) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    
                if self.prn: 
                    print("Client %d: predictor loss: %.2f | adversary loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, adv_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if epsilon: 
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

            if adaptive_lr: learning_rate = self.disparity(n_yz_c)/100

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

    def BiasCorrecting(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, alpha = 0.1, 
                    optimizer = "adam", epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        mu = torch.tensor([0.0,0.0]).type(torch.DoubleTensor)
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')
            
            n, nz, yz, yhat = 0, torch.tensor([0,0]), torch.tensor([0,0]), 0
            nc = []
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "bias correcting", seed = self.seed, prn = self.train_prn)
                n_, nz_, yz_, yhat_ = local_model.bc_compute(copy.deepcopy(self.model), 
                            mu)
                n, nz, yz, yhat = n + n_, nz + nz_, yz + yz_, yhat + yhat_
                nc.append(n_)

            delta = torch.tensor([0.0,0.0]).type(torch.DoubleTensor)
            delta[0], delta[1] = yz[0]/nz[0] - yhat/n, yz[1]/nz[1] - yhat/n
            mu = mu - alpha * delta

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "bias correcting", seed = self.seed, prn = self.train_prn)

                w, loss = local_model.bc_update(
                                model=copy.deepcopy(self.model), mu = mu, global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            # weights = average_weights(local_weights, self.clients_idx, idxs_users)
            weights = weighted_average_weights(local_weights, nc, n)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
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
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

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

    def Zafar(self, test_rounds = 3, test_lr = 0.005, test_penalty = 100, num_rounds = 4, local_epochs = 30, learning_rate = 0.0001, penalty = 50, optimizer = 'adam'):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # compute mean value of the sensitive attribute
        sum_z, len_z = 0, 0
        for c in range(self.num_clients):
            local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)
            sum_z_, len_z_ = local_model.mean_sensitive_stat()
            sum_z, len_z = sum_z + sum_z_, len_z + len_z_
        mean_z = sum_z / len_z

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        init_weights = copy.deepcopy(self.model).state_dict()

        model_states = []
        dp = []
        
        # choose the side of constraint
        for left in [True, False]:
            dp_ = []
            self.model.load_state_dict(init_weights)

            for round_ in tqdm(range(test_rounds)):
                local_weights, local_losses = [], []
                constraint = ['> -c', '< c'][int(left)]
                if self.prn: print(f'\n | Testing Round : {round_+1} | constraint :  Cov(z, d) {constraint}\n')

                self.model.train()
                m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
                idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

                for idx in idxs_users:
                    local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                                batch_size = self.batch_size, option = "zafar", seed = self.seed, prn = self.train_prn, penalty = test_penalty)

                    w, loss = local_model.zafar_update(
                                    model=copy.deepcopy(self.model), global_round=round_, 
                                        learning_rate = test_lr, local_epochs = local_epochs, 
                                        optimizer = optimizer, mean_z = mean_z, left = left)
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
                    local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                                batch_size = self.batch_size, option = "zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)
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
                        print(f' \nAvg Training Stats after {round_+1} Testing rounds:')
                        print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                            np.mean(np.array(train_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_yz)))
                dp_.append(self.disparity(n_yz))
            
            dp.append(sum(dp_))
            model_states.append(copy.deepcopy(self.model).state_dict())

        left = True if dp[0] < dp[1] else False
        self.model.load_state_dict(model_states[1-int(left)])

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            constraint = ['> -c', '< c'][int(left)]
            if self.prn: print(f'\n | Global Round : {round_+1} | constraint :  Cov(z, d) {constraint}\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)

                w, loss = local_model.zafar_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, mean_z = mean_z, left = left)
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
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                            batch_size = self.batch_size, option = "zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)
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

    def FTrain(self, num_rounds = 10, local_epochs = 30, lr_g = 0.005, lr_d = 0.01, 
                 epsilon = None, sensitive_level = 2, num_classes = 2, ratio_gd = 3,
                 lambda_d = 0.2, init_epochs = 100):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # discriminator: estimate the sensitive attribute
        discriminator = logReg(num_classes, sensitive_level)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        
        for round_ in tqdm(range(num_rounds)):
            local_weights_g, local_weights_d, local_losses = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_client = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "ftrain", seed = self.seed, prn = self.train_prn)

                w_g, w_d, loss = local_client.ftrain_update(
                                generator = copy.deepcopy(self.model), discriminator = discriminator, global_round=round_, 
                                    lr_g = lr_g, lr_d = lr_d, local_epochs = local_epochs, ratio_gd = ratio_gd, 
                                     lambda_d = lambda_d, init_epochs = init_epochs)
                local_weights_g.append(copy.deepcopy(w_g))
                local_weights_d.append(copy.deepcopy(w_d))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights_g = average_weights(local_weights_g, self.clients_idx, idxs_users)
            weights_d = average_weights(local_weights_d, self.clients_idx, idxs_users)
            self.model.load_state_dict(weights_g)
            discriminator.load_state_dict(weights_d)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_client = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn)
                # validation dataset inference
                acc, n_yz_c, acc_loss, adv_loss = local_client.al_inference(self.model, discriminator) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    
                if self.prn: 
                    print("Client %d: predictor loss: %.2f | adversary loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, adv_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if epsilon: 
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

            # if adaptive_lr: learning_rate = self.disparity(n_yz_c)/100

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

    # pre-processing
    def preBC(self, pre_rounds = 8, num_rounds = 2, local_epochs = 30, alpha = 0.01, learning_rate = 0.005, 
                    optimizer = "adam", epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        w0_z = [1, 1]
        w_yz = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
        p, delta, lbd = [0,0], [0,0], [0,0]
        n_yz, n_yz_c = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}, []
        for c in range(self.num_clients):
            n_yz_c.append(Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "prebc").get_n_yz())
            for yz in n_yz_c[c]:
                n_yz[yz] = n_yz[yz] + n_yz_c[c][yz]
        nw_yz = copy.deepcopy(n_yz)

        for round_ in range(pre_rounds):
            for yz in n_yz:
                nw_yz[yz] = n_yz[yz] * w_yz[yz]
            p[0] = w_yz[(1,0)] * nw_yz[(1,0)] / (w_yz[(1,0)] * nw_yz[(1,0)] + w_yz[(0,0)] * nw_yz[(0,0)])
            p[1] = w_yz[(1,1)] * nw_yz[(1,1)] / (w_yz[(1,1)] * nw_yz[(1,1)] + w_yz[(0,1)] * nw_yz[(0,1)])
            nw0, nw1, nw = nw_yz[(1,0)] + nw_yz[(0,0)], nw_yz[(1,1)] + nw_yz[(0,1)], nw_yz[(1,0)] + nw_yz[(0,0)] + nw_yz[(1,1)] + nw_yz[(0,1)]
            delta[0] = p[0] * (1 - nw0/nw) - nw1/nw * p[1]
            delta[1] = p[1] * (1 - nw1/nw) - nw0/nw * p[0]
            print(delta)
            lbd[0] = - alpha * delta[0]
            lbd[1] = - alpha * delta[1]
            for z in [0,1]: w0_z[z] = w0_z[z] * np.exp(lbd[z])
            for yz in w_yz:
                if yz[0] == 1: 
                    w_yz[yz] = w0_z[yz[1]] / (w0_z[yz[1]] + 1)
                else:
                    w_yz[yz] = 1 / (w0_z[yz[1]] + 1)

        mu = torch.tensor(w0_z).type(torch.DoubleTensor)
        print(mu)
        nc = []
        for c in range(self.num_clients):
            nc.append(0)
            for yz in n_yz_c[c]:
                n_yz_c[c][yz] = n_yz_c[c][yz] * w_yz[yz]
                nc[-1] += n_yz_c[c][yz]
        n = sum(nc)

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')


            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "prebc", seed = self.seed, prn = self.train_prn)

                w, loss = local_model.bc_update(
                                model=copy.deepcopy(self.model), mu = mu, global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            # weights = average_weights(local_weights, self.clients_idx, idxs_users)
            weights = weighted_average_weights(local_weights, nc, n)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
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
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

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

    def BCVariant1(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, alpha = 0.1, 
                    optimizer = "adam", epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        mu = torch.tensor([0.0,0.0]).type(torch.DoubleTensor)
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')
            
            n, nz, yz, yhat = 0, torch.tensor([0,0]), torch.tensor([0,0]), 0
            nc = []
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "bc-variant1", seed = self.seed, prn = self.train_prn)
                n_, nz_, yz_, yhat_ = local_model.bc1_compute(copy.deepcopy(self.model), 
                            mu)
                n, nz, yz, yhat = n + n_, nz + nz_, yz + yz_, yhat + yhat_
                nc.append(n_)

            delta = torch.tensor([0.0,0.0]).type(torch.DoubleTensor)
            delta[0], delta[1] = yz[0]/nz[0] - yhat/n, yz[1]/nz[1] - yhat/n
            mu = mu - alpha * delta

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "bc-variant1", seed = self.seed, prn = self.train_prn)

                w, loss = local_model.bc_update(
                                model=copy.deepcopy(self.model), mu = mu, global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            # weights = average_weights(local_weights, self.clients_idx, idxs_users)
            weights = weighted_average_weights(local_weights, nc, n)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
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
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

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

    def BCVariant2(self, num_rounds = 1, local_epochs = 30, learning_rate = 0.005, bs_iter = 5, 
                    optimizer = "adam", epsilon = None, lbd_interval = [[-10, 10], [-10, 10]]):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        lbd_interval = torch.tensor(lbd_interval).type(torch.DoubleTensor)

        # initialize the weights for each client
        nc = list(map(len, self.clients_idx))
        n = sum(nc)

        for _ in tqdm(range(bs_iter)):

            for round_ in range(num_rounds):
                local_weights, local_losses = [], []
                if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')
                
                lbd = torch.mean(lbd_interval, dim = 1)
                
                self.model.train()
                m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
                idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

                for idx in idxs_users:
                    local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                                batch_size = self.batch_size, option = "bc-variant2", seed = self.seed, prn = self.train_prn)

                    w, loss = local_model.bc_update(
                                    model=copy.deepcopy(self.model), mu = lbd, global_round=round_, 
                                        learning_rate = learning_rate, local_epochs = local_epochs, 
                                        optimizer = optimizer)

                    local_weights.append(copy.deepcopy(w))
                    local_losses.append(copy.deepcopy(loss))

                # update global weights
                # weights = average_weights(local_weights, self.clients_idx, idxs_users)
                weights = weighted_average_weights(local_weights, nc, n)
                self.model.load_state_dict(weights)

                loss_avg = sum(local_losses) / len(local_losses)
                train_loss.append(loss_avg)

                n, nz, yz, yhat = 0, torch.tensor([0,0]), torch.tensor([0,0]), 0
                nc = []
                for c in range(self.num_clients):
                    local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                                batch_size = self.batch_size, option = "bias correcting", seed = self.seed, prn = self.train_prn)
                    n_, nz_, yz_, yhat_ = local_model.bc_compute(copy.deepcopy(self.model), 
                                lbd)
                    n, nz, yz, yhat = n + n_, nz + nz_, yz + yz_, yhat + yhat_
                    nc.append(n_)

                delta = torch.tensor([0.0,0.0]).type(torch.DoubleTensor)
                delta[0], delta[1] = yz[0]/nz[0] - yhat/n, yz[1]/nz[1] - yhat/n

                for z in [0,1]:
                    if delta[z] <= 0:
                        lbd_interval[z][0] = lbd[z]
                    elif delta[z] >= 0:
                        lbd_interval[z][1] = lbd[z]


                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
                self.model.eval()
                for c in range(m):
                    local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
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
                    if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

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

    def FBVariant1(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
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
            (0,0): m_yz[(0,0)]/(m_yz[(1,0)] + m_yz[(0,0)]), 
            (0,1): m_yz[(0,1)]/(m_yz[(0,1)] + m_yz[(1,1)]),
            (1,0): m_yz[(1,0)]/(m_yz[(1,0)] + m_yz[(0,0)]),
            (1,1): m_yz[(1,1)]/(m_yz[(0,1)] + m_yz[(1,1)]),
        }

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses, nc = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                        option = "FB-Variant1", lbd = lbd, 
                                        seed = self.seed, prn = self.train_prn)

                w, loss, nc_ = local_model.fb_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, lbd = lbd, m_yz = m_yz)
                nc.append(nc_)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nc, sum(nc))
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

            y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
            y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
            if y0_diff > y1_diff:
                lbd[(0,0)] -= alpha / (round_+1)
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(1,0)] = 1 - lbd[(0,0)]
                lbd[(0,1)] += alpha / (round_+1)
                lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                lbd[(1,1)] = 1 - lbd[(0,1)]
            else:
                lbd[(0,0)] += alpha / (round_+1)
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(0,1)] = 1 - lbd[(0,0)]
                lbd[(1,0)] -= alpha / (round_+1)
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

    def BCVariant3(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, alpha = 0.1, 
                    optimizer = "adam", epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        mu = torch.tensor([0.0,0.0]).type(torch.DoubleTensor)
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')
            
            n, nz, yz, yhat = 0, torch.tensor([0,0]), torch.tensor([0,0]), 0
            nc = []
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "bias correcting", seed = self.seed, prn = self.train_prn)
                n_, nz_, yz_, yhat_ = local_model.bc_compute(copy.deepcopy(self.model), 
                            mu)
                n, nz, yz, yhat = n + n_, nz + nz_, yz + yz_, yhat + yhat_
                nc.append(n_)

            delta = torch.tensor([0.0,0.0]).type(torch.DoubleTensor)
            delta[0], delta[1] = yz[0]/nz[0] - yhat/n, yz[1]/nz[1] - yhat/n
            mu = mu - alpha * delta

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "bias correcting", seed = self.seed, prn = self.train_prn)

                w, loss = local_model.bc3_update(
                                model=copy.deepcopy(self.model), mu = mu, global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            # weights = average_weights(local_weights, self.clients_idx, idxs_users)
            weights = weighted_average_weights(local_weights, nc, n)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {(0,0):0, (0,1):0, (1,0):0, (1,1):0}
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
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
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

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

    def FBVariant2(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 1, alpha_decay = 0.5):
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
            (0,0): m_yz[(0,0)]/(m_yz[(1,0)] + m_yz[(0,0)]), 
            (0,1): m_yz[(0,1)]/(m_yz[(0,1)] + m_yz[(1,1)]),
            (1,0): m_yz[(1,0)]/(m_yz[(1,0)] + m_yz[(0,0)]),
            (1,1): m_yz[(1,1)]/(m_yz[(0,1)] + m_yz[(1,1)]),
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

            y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
            y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
            if y0_diff > y1_diff:
                lbd[(0,0)] -= alpha / (round_+1) ** .5 
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(1,0)] = 1 - lbd[(0,0)]
                lbd[(0,1)] += alpha / (round_+1) ** .5 
                lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                lbd[(1,1)] = 1 - lbd[(0,1)]
            else:
                lbd[(0,0)] += alpha / (round_+1) ** .5 
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(0,1)] = 1 - lbd[(0,0)]
                lbd[(1,0)] -= alpha / (round_+1) ** .5 
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

            alpha = alpha_decay * alpha

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