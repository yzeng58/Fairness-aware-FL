import torch, copy, time, random, sys

import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '..')
from utils import *
from fairBatch import *

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#########################################################

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False, 
                train_prn = False, batch_size = 128, print_every = 1, Z = 2):
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

        self.train_dataset, self.test_dataset, _ = dataset_info
        self.trainloader, self.validloader = self.train_val(self.train_dataset, batch_size)
        self.Z = Z

    def train_val(self, dataset, batch_size, idxs_train_full = None, split = False):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)
        
        # split indexes for train, validation (90, 10)
        if idxs_train_full == None: idxs_train_full = np.arange(len(dataset))
        idxs_train = idxs_train_full[:int(0.9*len(idxs_train_full))]
        idxs_val = idxs_train_full[int(0.9*len(idxs_train_full)):]
    
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                    batch_size=batch_size, shuffle=True)
        if split:
            validloader = {}
            for sen in range(self.Z):
                sen_idx = np.where(dataset.sen[idxs_train_full] == sen)[0]
                validloader[sen] = DataLoader(DatasetSplit(dataset, idxs_train_full[sen_idx]),
                                        batch_size=max(int(len(idxs_train_full)/10),10), shuffle=False)
        else:
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def Unconstrained(self, num_epochs = 30, learning_rate = 0.005, 
                    optimizer = "adam", epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)

        # Training
        train_loss = []
        start_time = time.time()
        
        for round_ in tqdm(range(num_epochs)):
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            batch_loss = []
            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs
                
                probas, logits = self.model(features)
                loss, _, _ = loss_func('unconstrained', logits, labels, probas, sensitive, 0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            # update global weights
            train_loss.append(sum(batch_loss)/len(batch_loss))

            # Calculate avg training accuracy over all clients at every round
            # the number of samples which are assigned to class y and belong to the sensitive group z
            self.model.eval()
            acc, loss, n_eyz, acc_loss, fair_loss, _ = self.inference() 
                    
            if self.prn: 
                print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                        acc_loss, fair_loss, self.metric, self.disparity(n_eyz)))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nTraining Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*acc, self.metric, self.disparity(n_eyz)))

            if epsilon: 
                if self.disparity(n_eyz) < epsilon and acc > 0.5: break

        # Test inference after completion of training
        test_acc, n_eyz= self.test_inference()
        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Train Accuracy: {:.2f}%".format(100*acc))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    def FairConstraints(self, num_epochs = 30, learning_rate = 0.001, penalty = 500, optimizer = 'adam', epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        
        for round_ in tqdm(range(num_epochs)):
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            batch_loss = []
            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                
                _, logits = self.model(features)
                loss = eo_loss(logits,labels, sensitive, penalty)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            train_loss.append(sum(batch_loss)/len(batch_loss))

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            self.model.eval()
            acc, loss, n_eyz, acc_loss, fair_loss, _ = self.inference(model = self.model, train = True) 
            list_acc.append(acc)
                
            if self.prn: 
                print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                        acc_loss, fair_loss, self.metric, self.disparity(n_eyz)))

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
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    def FairBatch(self, num_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3, trace = False):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()

        # the number of samples whose label is y and sensitive attribute is z
        m_1z = []
        for z in range(self.Z):
            m_1z.append(((self.train_dataset.y == 1) & (self.train_dataset.sen == z)).sum())

        m = len(self.train_dataset.y)
        lbd = []
        for z in range(self.Z):
            lbd.append(m_1z[z]/len(self.train_dataset.y))

        if trace: acc_l, dp_l = [], []
        for round_ in tqdm(range(num_epochs)):
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            batch_loss = []
            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                _, logits = self.model(features)

                v = torch.randn(len(labels)).type(torch.DoubleTensor)
                
                group_idx = {}
                
                for z in range(self.Z):
                    group_idx[(0,z)] = torch.where((labels == 0) & (sensitive == z))[0]
                    group_idx[(1,z)] = torch.where((labels == 1) & (sensitive == z))[0]
                    v[group_idx[(0,z)]] = 1
                    v[group_idx[(1,z)]] = lbd[z] * m / m_1z[z]

                loss = weighted_loss(logits, labels, v)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            train_loss.append(sum(batch_loss)/len(batch_loss))

            # Calculate avg training accuracy over all clients at every round
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_eyz, loss_yz = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    loss_yz[(y,z)] = 0
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0

            self.model.eval()
            acc, loss, n_eyz, acc_loss, fair_loss, loss_yz = self.inference(option = "FairBatch", model = self.model, train = True) 
            train_accuracy.append(acc)
                
            if self.prn: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                acc_loss, fair_loss, self.metric, self.disparity(n_eyz)))
                
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


            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))

            if trace and ((round_+1) % trace == 0) and (round_ >= 200):
                test_acc, n_yz = self.test_inference(self.model, self.test_dataset)
                rd = self.disparity(n_yz)
                acc_l.append(test_acc)
                dp_l.append(rd)

        # Test inference after completion of training
        test_acc, n_eyz = self.test_inference(self.model, self.test_dataset)
        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: 
            if trace:
                return acc_l, dp_l
            else:
                return test_acc, rd

    def BiasCorrecting(self, num_epochs = 30, learning_rate = 0.005, alpha = 0.1, 
                    optimizer = "adam"):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_accuracy = []
        start_time = time.time()
        mu = torch.zeros(self.Z).type(torch.DoubleTensor)
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)

        pg = torch.zeros(self.Z).type(torch.DoubleTensor)
        for z in range(self.Z):
            pg[z] = sum(self.train_dataset.y * (self.train_dataset.sen == z)) / len(self.train_dataset.y)
        px = sum(self.train_dataset.y) / len(self.train_dataset.y)
        
        for round_ in tqdm(range(num_epochs)):
            delta = torch.zeros(self.Z).type(torch.DoubleTensor)
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')
            
            x, y, z = torch.tensor(self.train_dataset.x), torch.tensor(self.train_dataset.y), torch.tensor(self.train_dataset.sen)
            x = x.to(DEVICE)
            probas, _ = self.model(x)
            probas = probas.T[1] / torch.sum(probas, dim = 1)
            delta = []
            for z_ in range(self.Z):
                delta.append(sum(probas * ((z == z_) * y / pg[z_] - y / px)))
                delta[z_].detach()

            delta = torch.tensor(delta).type(torch.DoubleTensor)
            delta = delta / len(self.train_dataset.y)
            mu = mu - alpha * delta # / np.sqrt(round_ + 1)

            self.model.train()
            batch_loss = []
            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                
                v = torch.randn(len(labels)).type(torch.DoubleTensor)
                # labels_i == 1
                y1_idx = torch.where(labels == 1)[0]
                exp = np.exp(torch.index_select(mu, 0, sensitive[y1_idx].type(torch.LongTensor)))
                v[y1_idx] = exp / (1 + exp)

                # labels_i == 0
                y0_idx = torch.where(labels == 0)[0].type(torch.LongTensor)
                exp = np.exp(torch.index_select(mu, 0, sensitive[y0_idx].type(torch.LongTensor)))
                v[y0_idx] = 1 / (1 + exp)

                _, logits = self.model(features)
                loss = weighted_loss(logits, labels, v)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            self.model.eval()
            acc, loss, n_eyz, acc_loss, fair_loss, _ = self.inference(model = self.model) 
            list_acc.append(acc)
        
            if self.prn: 
                print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                         acc_loss, fair_loss, self.metric, self.disparity(n_eyz)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(batch_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))


        # Test inference after completion of training
        test_acc, n_eyz= self.test_inference()
        rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd  

    def FTrain(self, num_epochs = 30, lr_g = 0.005, lr_d = 0.01, epsilon = None, num_classes = 2, ratio_gd = 3,
                    lambda_d = 0.2, init_epochs = 100):
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # discriminator: estimate the sensitive attribute
            discriminator = logReg(num_classes, self.Z)
            generator = self.model 

            optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr_g)
            optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = lr_d)

            # Training
            train_accuracy = []
            start_time = time.time()
            
            for epoch in tqdm(range(num_epochs)):
                if self.prn: print(f'\n | Global Training Round : {epoch+1} |\n')

                self.model.train()    
                batch_loss = [] 
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE).type(torch.LongTensor)

                    # for the first 100 epochs, train both generator and discriminator
                    # after the first 100 epochs, ratio of updating generator and discriminator (1:ratio_gd training)
                    if epoch % ratio_gd == 0 or epoch < init_epochs:
                        # forward generator
                        optimizer_G.zero_grad()
                    
                    y1_idx = torch.where(labels == 1)
                    _, logits_g = generator(features)

                    # train fairness discriminator
                    optimizer_D.zero_grad()
                    loss_d = F.cross_entropy(discriminator(logits_g[y1_idx].detach())[1], sensitive[y1_idx])
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
                    batch_loss.append(loss_g.item())

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z

                self.model.eval()
                acc, n_eyz, acc_loss, adv_loss = self.al_inference(self.model, discriminator) 
                list_acc.append(acc)
                        
                if self.prn: 
                    print("Predictor loss: %.2f | adversary loss %.2f | %s = %.2f" % (
                            acc_loss, adv_loss, self.metric, self.disparity(n_eyz)))

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn:
                    if (epoch+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                        print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                            np.mean(np.array(batch_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_eyz)))

                if epsilon: 
                    if self.disparity(n_eyz) < epsilon and train_accuracy[-1] > 0.5: break

                # if adaptive_lr: learning_rate = self.disparity(n_eyz_c)/100

            # Test inference after completion of training
            test_acc, n_eyz= self.test_inference()
            rd = self.disparity(n_eyz)

            if self.prn:
                print(f' \n Results after {num_epochs} global rounds of training:')
                print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
                print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

                # Compute fairness metric
                print("|---- Test "+ self.metric+": {:.4f}".format(rd))

                print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

            if self.ret: return test_acc, rd

    # post-processing approach
    def ThresholdAdjust(self, num_epochs = 30, learning_rate = 0.005, epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        start_time = time.time()
        models = [copy.deepcopy(self.model) for _ in range(self.Z)]

        idx_sen = {}
        _, validloader = self.train_val(self.train_dataset, self.batch_size, split = True)
        for sen in range(self.Z):
            idx_sen[sen] = np.arange(len(self.train_dataset))[self.train_dataset.sen == sen]

        _, n_eyz = self.test_inference()
        for round_ in tqdm(range(num_epochs)):
            eod = EODisparity(n_eyz, True)
            if self.disparity(n_eyz) <= epsilon: break
            
            min_z, max_z = np.argmin(eod), np.argmax(eod)
            update_step = {min_z:1, max_z:-1}
            z = [min_z, max_z]

            optimizer = [torch.optim.Adam(models[z[0]].parameters(), lr=0.005, weight_decay=1e-4),
                        torch.optim.Adam(models[z[1]].parameters(), lr=0.005, weight_decay=1e-4)]

            models[z[0]].train()
            models[z[1]].train()

            param0 = next(models[z[0]].parameters())
            param0.requires_grad = False

            param1 = next(models[z[1]].parameters())
            param1.requires_grad = False

            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                
                group_idx, loss, bias_grad, w = [], [0,0], [0,0], []
                for i in [0,1]:
                    w.append(copy.deepcopy(models[z[i]]).state_dict())
                    group_idx.append(torch.where(sensitive == z[i]))

                    probas, logits = models[z[i]](features[group_idx[i]])
                    loss[i], _, _ = loss_func('threshold adjusting', logits, labels[group_idx[i]], probas, sensitive[group_idx[i]])
                    optimizer[i].zero_grad()
                    # compute the gradients
                    loss[i].backward()
                    # get the gradient of the bias
                    bias_grad[i] = models[z[0]].linear.bias.grad[1]

                update_idx = np.argmin(bias_grad)
                w[update_idx]['linear.bias'][1] += learning_rate * update_step[z[update_idx]] / (round_ + 1) ** .5
                w[update_idx]['linear.bias'][0] -= learning_rate * update_step[z[update_idx]] / (round_ + 1) ** .5
                models[z[update_idx]].load_state_dict(w[update_idx])

            n_eyz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    for e in [0,1]:
                        n_eyz[(e,y,z)] = 0

            train_loss, train_acc = [], []
            for sen in range(self.Z):
                acc, loss, n_eyz_c, _, _, _ = self.inference(model = models[sen], validloader = validloader[sen])
                train_loss.append(loss)
                train_acc.append(acc)

                for eyz in n_eyz:
                    n_eyz[eyz] += n_eyz_c[eyz]            

            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} threshold adjusting global rounds:')
                    print("Training loss: %.2f | Train accuracy: %.2f%% | Train %s: %.4f" % (
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
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    def test_inference(self, model = None, test_dataset = None):
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

    def inference(self, option = 'unconstrained', penalty = 100, model = None, validloader = None, train = False):
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

        if model == None: model = self.model
        if validloader == None: 
            validloader = self.validloader
        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_eyz, loss_yz = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                for e in [0,1]:
                    n_eyz[(e,y,z)] = 0
        
        dataset = self.trainloader if train else validloader
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
            
            for yz in loss_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                if option == "FairBatch":
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("FairBatch", logits[group_boolean_idx[yz]], 
                                                    labels[group_boolean_idx[yz]], 
                                         outputs[group_boolean_idx[yz]], sensitive[group_boolean_idx[yz]], 
                                         penalty)
                    loss_yz[yz] += loss_yz_

            for e,y,z in n_eyz:
                n_eyz[(e,y,z)] += torch.sum((pred_labels == e) & (sensitive == z) & (labels == y)).item()     
                
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(option, logits, 
                                                        labels, outputs, sensitive, penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if option in ["FairBatch", "FB-Variant1"]:
            return accuracy, loss, n_eyz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_eyz, acc_loss / num_batch, fair_loss / num_batch, None

    def al_inference(self, model, adv_model):
        model.eval()
        total, correct, acc_loss, adv_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0
        n_eyz = {}
        for y in [0,1]:
            for z in range(self.Z):
                for e in [0,1]:
                    n_eyz[(e,y,z)] = 0
        
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
            
            for eyz in n_eyz:
                group_boolean_idx[eyz] = (pred_labels == eyz[0]) & (labels == eyz[1]) & (sensitive == eyz[2])
                n_eyz[eyz] += torch.sum(group_boolean_idx[eyz]).item()     
            
            batch_acc_loss, batch_adv_loss = al_loss(logits, labels, adv_logits, sensitive)
            acc_loss, adv_loss = (acc_loss + batch_acc_loss.item(), 
                                         adv_loss + batch_adv_loss.item())
        accuracy = correct/total
        return accuracy, n_eyz, acc_loss / num_batch, adv_loss / num_batch