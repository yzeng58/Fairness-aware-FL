# Server for equal opportunity

import torch, copy, time, random, sys

import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict

from Client import *

sys.path.insert(0, '..')
from utils import *
from fairBatch import *

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#########################################################

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False, 
                train_prn = False, batch_size = 128, print_every = 1, fraction_clients = 1, Z = 2):
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

        self.metric = "Equal Opportunity Disparity"
        self.disparity = EODisparity

        self.batch_size = batch_size
        self.print_every = print_every
        self.fraction_clients = fraction_clients

        self.train_dataset, self.test_dataset, self.clients_idx = dataset_info
        self.num_clients = len(self.clients_idx)
        self.Z = Z

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
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn,
                            Z = self.Z)

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
            n_eyz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0

            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn,
                            Z = self.Z)
                # validation dataset inference
                acc, loss, n_eyz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for e,y,z in n_eyz:
                    n_eyz[(e,y,z)] += n_eyz_c[(e,y,z)]
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_eyz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))
            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))

            if epsilon: 
                if self.disparity(n_eyz) < epsilon and train_accuracy[-1] > 0.5: break

        # Test inference after completion of training
        test_acc, n_eyz= self.test_inference()
        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    def FairBatch(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        # the number of samples whose label is y and sensitive attribute is z
        m_1z = []
        for z in range(self.Z):
            m_1z.append(((self.train_dataset.y == 1) & (self.train_dataset.sen == z)).sum())

        lbd = []
        for z in range(self.Z):
            lbd.append(m_1z[z]/len(self.train_dataset.y))

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses, nc = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                        option = "Unconstrained", lbd = lbd, 
                                        seed = self.seed, prn = self.train_prn,
                                        Z = self.Z)

                w, loss, nc_ = local_model.fb_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate / np.sqrt(round_ + 1), local_epochs = local_epochs, 
                                    optimizer = optimizer, lbd = lbd, m_1z = m_1z, m = len(self.train_dataset.y))
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
            n_eyz, loss_yz = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    loss_yz[(y,z)] = 0
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0

            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FairBatch", 
                                            lbd = lbd, seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_eyz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for e,y,z in n_eyz:
                    n_eyz[(e,y,z)] += n_eyz_c[(e,y,z)]
                    loss_yz[(y,z)] += loss_yz_c[(y,z)]
                    
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    c+1, acc_loss, fair_loss, self.metric, self.disparity(n_eyz_c)))
                
            max_bias, z_selected = 0, 0
            for z in range(self.Z):
                z_ = (z + 1) % self.Z
                bias_z = abs(loss_yz[(1,z)]/m_1z[z] - loss_yz[(1,z_)]/m_1z[z_])
                if bias_z > max_bias:
                    if loss_yz[(1,z)]/m_1z[z] > loss_yz[(1,z_)]/m_1z[z_]:
                        z_selected = z
                    else:
                        z_selected = z_
                    max_bias = bias_z

            for z in range(self.Z):
                if z == z_selected:
                    lbd[z_selected] += alpha / (round_ + 1)
                else:
                    m1z_ = copy.deepcopy(m_1z)
                    m1z_.pop(z_selected)
                    lbd[z] -= alpha / (round_ + 1) * m_1z[z] / sum(m1z_)

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))

        # Test inference after completion of training
        test_acc, n_eyz = self.test_inference(self.model, self.test_dataset)
        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    def BiasCorrecting(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, alpha = 0.1, 
                    optimizer = "adam"):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        mu = torch.zeros(self.Z).type(torch.DoubleTensor)

        pg = torch.zeros(self.Z).type(torch.DoubleTensor)
        for z in range(self.Z):
            pg[z] = sum(self.train_dataset.y * (self.train_dataset.sen == z)) / len(self.train_dataset.y)
        px = sum(self.train_dataset.y) / len(self.train_dataset.y)
        
        for round_ in tqdm(range(num_rounds)):
            delta = torch.zeros(self.Z).type(torch.DoubleTensor)
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')
            
            nc = []
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "bias correcting", seed = self.seed, prn = self.train_prn)
                delta_ = local_model.bc_compute(copy.deepcopy(self.model), px, pg)
                delta += delta_

            delta = delta / len(self.train_dataset.y)
            mu = mu - alpha * delta # / np.sqrt(round_ + 1)

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "bias correcting", seed = self.seed, prn = self.train_prn)

                w, loss, nc_ = local_model.bc_update(
                                model=copy.deepcopy(self.model), mu = mu, global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                nc.append(nc_)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            # weights = average_weights(local_weights, self.clients_idx, idxs_users)
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_eyz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn)
                # validation dataset inference
                acc, loss, n_eyz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for eyz in n_eyz:
                    n_eyz[eyz] += n_eyz_c[eyz]
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_eyz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))


        # Test inference after completion of training
        test_acc, n_eyz= self.test_inference()
        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd  

    def FairConstraints(self, test_rounds = 3, test_lr = 0.005, test_penalty = 100, num_rounds = 4, local_epochs = 30, learning_rate = 0.0001, penalty = 50, optimizer = 'adam'):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # compute mean value of the sensitive attribute
        mean_z1 = self.train_dataset.sen[self.train_dataset.y == 1].mean().item()

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
                if self.prn: print(f'\n | Testing Round : {round_+1} | constraint :  Cov(z, d | y = 1) {constraint}\n')

                self.model.train()
                m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
                idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

                for idx in idxs_users:
                    local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                                batch_size = self.batch_size, option = "eofc", seed = self.seed, prn = self.train_prn, penalty = test_penalty)

                    w, loss = local_model.fc_update(
                                    model=copy.deepcopy(self.model), global_round=round_, 
                                        learning_rate = test_lr, local_epochs = local_epochs, 
                                        optimizer = optimizer, mean_z1 = mean_z1, left = left)
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
                n_eyz = {}
                for y in [0,1]:
                    for z in range(self.Z):
                        for e in [0,1]:
                            n_eyz[(e,y,z)] = 0
                self.model.eval()
                for c in range(m):
                    local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                                batch_size = self.batch_size, option = "eofc", seed = self.seed, prn = self.train_prn, penalty = penalty)
                    # validation dataset inference
                    acc, loss, n_eyz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model) 
                    list_acc.append(acc)
                    
                    for eyz in n_eyz:
                        n_eyz[eyz] += n_eyz_c[eyz]
                        
                    if self.prn: 
                        print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                                c+1, acc_loss, fair_loss, self.metric, self.disparity(n_eyz_c)))

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} Testing rounds:')
                        print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                            np.mean(np.array(train_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))
                dp_.append(self.disparity(n_eyz))
            
            dp.append(sum(dp_))
            model_states.append(copy.deepcopy(self.model).state_dict())

        left = True if dp[0] < dp[1] else False
        self.model.load_state_dict(model_states[1-int(left)])

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            constraint = ['> -c', '< c'][int(left)]
            if self.prn: print(f'\n | Global Round : {round_+1} | constraint :  Cov(z, d | y = 1) {constraint}\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)

                w, loss = local_model.fc_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, mean_z1 = mean_z1, left = left)
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
            n_eyz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                            batch_size = self.batch_size, option = "zafar", seed = self.seed, prn = self.train_prn, penalty = penalty)
                # validation dataset inference
                acc, loss, n_eyz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for eyz in n_eyz:
                    n_eyz[eyz] += n_eyz_c[eyz]
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_eyz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))
            
        # Test inference after completion of training
        test_acc, n_eyz= self.test_inference()
        rd = self.disparity(n_eyz)

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
                n_eyz = {}
                for y in [0,1]:
                    for z in range(self.Z):
                        for e in [0,1]:
                            n_eyz[(e,y,z)] = 0

                self.model.eval()
                for c in range(m):
                    local_client = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                                batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn)
                    # validation dataset inference
                    acc, n_eyz_c, acc_loss, adv_loss = local_client.al_inference(self.model, discriminator) 
                    list_acc.append(acc)
                    
                    for eyz in n_eyz:
                        n_eyz[eyz] += n_eyz_c[eyz]
                        
                    if self.prn: 
                        print("Client %d: predictor loss: %.2f | adversary loss %.2f | %s = %.2f" % (
                                c+1, acc_loss, adv_loss, self.metric, self.disparity(n_eyz_c)))

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} global rounds:')
                        print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                            np.mean(np.array(train_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))

                if epsilon: 
                    if self.disparity(n_eyz) < epsilon and train_accuracy[-1] > 0.5: break

                # if adaptive_lr: learning_rate = self.disparity(n_eyz_c)/100

            # Test inference after completion of training
            test_acc, n_eyz= self.test_inference()
            rd = self.disparity(n_eyz)

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
            n_eyz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0

            self.model.eval()
            for c in range(m):
                local_client = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn)
                # validation dataset inference
                acc, n_eyz_c, acc_loss, adv_loss = local_client.al_inference(self.model, adv_model) 
                list_acc.append(acc)
                
                for eyz in n_eyz:
                    n_eyz[eyz] += n_eyz_c[eyz]
                    
                if self.prn: 
                    print("Client %d: predictor loss: %.2f | adversary loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, adv_loss, self.metric, self.disparity(n_eyz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))

            if epsilon: 
                if self.disparity(n_eyz) < epsilon and train_accuracy[-1] > 0.5: break

            if adaptive_lr: learning_rate = self.disparity(n_eyz_c)/100

        # Test inference after completion of training
        test_acc, n_eyz= self.test_inference()
        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    # post-processing approach
    def ThresholdAdjust(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, 
                        epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        start_time = time.time()
        models = [copy.deepcopy(self.model) for _ in range(self.Z)]
        w = [[None for _ in range(self.num_clients)] for _ in range(2)]
        clients_idx_sen = defaultdict(list)
        

        for c in range(self.num_clients):
            for sen in range(self.Z):
                clients_idx_sen[sen].append(self.clients_idx[c][self.train_dataset.sen[self.clients_idx[c]] == sen])
        
        _, n_eyz = self.test_inference()
        for round_ in tqdm(range(num_rounds)):
            eod = EODisparity(n_eyz, True)
            if self.disparity(n_eyz) <= epsilon: break
            
            min_z, max_z = np.argmin(eod), np.argmax(eod)
            update_step = {min_z:1, max_z:-1}
            update_z = [min_z, max_z]

            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=np.concatenate((clients_idx_sen[min_z][c], clients_idx_sen[max_z][c]), axis = None), batch_size = self.batch_size, 
                                        option = "threshold adjusting",  
                                        seed = self.seed, prn = self.train_prn)

                w[0][c], w[1][c] = local_model.threshold_adjusting(
                                models=copy.deepcopy(models), 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = 'adam', update_step = update_step)
                
            # update global weights
            models[min_z].load_state_dict(average_weights(w[0], clients_idx_sen[update_z[0]], range(self.num_clients)))
            models[max_z].load_state_dict(average_weights(w[1], clients_idx_sen[update_z[1]], range(self.num_clients)))

            n_eyz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0

            train_loss, train_acc = [], []
            for c in range(self.num_clients):
                for sen in [0,1]:
                    local_model = Client(dataset=self.train_dataset,
                                                idxs=clients_idx_sen[sen][c], batch_size = self.batch_size, 
                                            option = "threshold adjusting",  
                                            seed = self.seed, prn = self.train_prn)
                    acc, loss, n_eyz_c, _, _, _ = local_model.inference(model = models[sen])
                    train_loss.append(loss)
                    train_acc.append(acc)

                    for eyz in n_eyz:
                        n_eyz[eyz] += n_eyz_c[eyz]            

            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} threshold adjusting global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*np.array(train_acc).mean(), self.metric, self.disparity(n_eyz)))
                
            if epsilon: 
                if self.disparity(n_eyz) < epsilon: break
  
        # Test inference after completion of training
        n_eyz = {}
        for y in [0,1]:
            for z in range(self.Z):
                for e in [0,1]:
                    n_eyz[(e,y,z)] = 0

        idx, test_acc_list = [], []
        for z in range(self.Z):
            idx.append(np.where(self.test_dataset.sen == z)[0].tolist())
            test_acc_, n_eyz_ = self.test_inference(models[z], DatasetSplit(self.test_dataset, idx[z]))
            test_acc_list.append(test_acc_)
            for e in [0,1]:
                for y in [0,1]:
                    n_eyz[(e,y,z)] = n_eyz_[(e,y,z)]
        
        test_acc = sum([test_acc_list[z] * len(idx[z]) for z in range(self.Z)]) / sum([len(idx[z]) for z in range(self.Z)])

        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
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
        n_eyz = {}
        for y in [0,1]:
            for z in range(self.Z):
                for e in [0,1]:
                    n_eyz[(e,y,z)] = 0
        
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
            
            for e,y,z in n_eyz:
                n_eyz[(e,y,z)] += torch.sum((pred_labels == e) & (sensitive == z) & (labels == y)).item()  

        accuracy = correct/total

        return accuracy, n_eyz