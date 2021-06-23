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

        self.disparity = DPDisparity
        self.metric = "Demographic disparity"

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
                sen_idx = np.where(dataset.sen[idxs_val] == sen)[0]
                validloader[sen] = DataLoader(DatasetSplit(dataset, idxs_val[sen_idx]),
                                        batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        else:
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def Unconstrained(self, num_epochs = 10, learning_rate = 0.005, 
                    optimizer = "adam", epsilon = None):
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
            local_losses = []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                
                probas, logits = self.model(features)
                loss, _, _ = loss_func("unconstrained", logits, labels, probas, sensitive, 100)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                local_losses.append(loss)

            train_loss.append(sum(local_losses)/len(local_losses))

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            self.model.eval()
            # validation dataset inference
            acc, loss, n_yz, acc_loss, fair_loss, _ = self.inference("unconstrained") 
            list_acc.append(acc)

                
            if self.prn: 
                print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                        acc_loss, fair_loss, self.metric, self.disparity(n_yz)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        sum(train_loss)/len(train_loss), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if epsilon: 
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

        # Test inference after completion of training
        test_acc, n_yz= self.test_inference()
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    # post-processing approach
    def ThresholdAdjust(self, num_epochs = 10, learning_rate = 0.7, 
                        epsilon = None, optimizer = 'adam'):
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
        _, n_yz = self.test_inference()
        for round_ in tqdm(range(num_epochs)):
            dpd = DPDisparity(n_yz, True)
            if self.disparity(n_yz) <= epsilon: break
            
            min_z, max_z = np.argmin(dpd), np.argmax(dpd)
            update_step = {min_z:1, max_z:-1}
            z = [min_z, max_z]

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
                    loss[i], _, _ = loss_func('threshold adjusting', logits, labels[group_idx[i]], probas, sensitive[group_idx[i]])
                    optimizer[i].zero_grad()
                    # compute the gradients
                    loss[i].backward()
                    # get the gradient of the bias
                    bias_grad[i] = models[z[0]].linear.bias.grad[1]

                update_idx = np.argmin(bias_grad)
                w[update_idx]['linear.bias'][1] += learning_rate * update_step[z[update_idx]] / (round_+1) ** .5
                w[update_idx]['linear.bias'][0] -= learning_rate * update_step[z[update_idx]] / (round_+1) ** .5
                models[z[update_idx]].load_state_dict(w[update_idx])

            n_yz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0

            train_loss, train_acc = [], []
            for sen in range(self.Z):
                acc, loss, n_yz_c, _, _, _ = self.inference(option = 'threshold adjusting', model = models[sen], validloader = validloader[sen])
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
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0

        idx, test_acc_list = [], []
        for z in range(self.Z):
            idx.append(np.where(self.test_dataset.sen == z)[0].tolist())
            test_acc_, n_yz_ = self.test_inference(models[z], DatasetSplit(self.test_dataset, idx[z]))
            test_acc_list.append(test_acc_)
            for y in [0,1]:
                n_yz[(y,z)] = n_yz_[(y,z)]
        
        test_acc = sum([test_acc_list[z] * len(idx[z]) for z in range(self.Z)]) / sum([len(idx[z]) for z in range(self.Z)])
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd
 
    def FairConstraints(self, num_epochs = 10, learning_rate = 0.001, penalty = 500, optimizer = 'adam', epsilon = None):
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
        train_accuracy = []
        start_time = time.time()
        
        for round_ in tqdm(range(num_epochs)):
            list_acc = []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            features, labels = torch.tensor(self.train_dataset.x).to(DEVICE), torch.tensor(self.train_dataset.y).to(DEVICE).type(torch.LongTensor)
            sensitive = torch.tensor(self.train_dataset.sen).to(DEVICE)
            probas, logits = self.model(features)
            loss, _, _ = loss_func('local zafar', logits, labels, probas, sensitive, penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # the number of samples which are assigned to class y and belong to the sensitive group z
            self.model.eval()

            acc, loss, n_yz, acc_loss, fair_loss, _ = self.inference("localFC", penalty) 
            list_acc.append(acc)
            train_accuracy.append(sum(list_acc)/len(list_acc))
                    

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            acc_loss, fair_loss, self.metric, self.disparity(n_yz)))
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        loss, 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if epsilon: 
                if self.disparity(n_yz) < epsilon and train_accuracy[-1] > 0.5: break

        # Test inference after completion of training
        test_acc, n_yz= self.test_inference()
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_epochs} global rounds of training:')
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
                            batch_size = self.batch_size, option = "al", seed = self.seed, prn = self.train_prn, Z = self.Z)

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
            n_yz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0

            self.model.eval()
            for c in range(m):
                local_client = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
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

    def BiasCorrecting(self, num_epochs = 30, learning_rate = 0.005, alpha = 0.1, optimizer = "adam", epsilon = None):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        mu = torch.zeros(self.Z).type(torch.DoubleTensor)
        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        
        for round_ in tqdm(range(num_epochs)):
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')
            self.model.eval()

            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            idxs = np.arange(len(self.train_dataset))
            idxs_train = idxs[:int(0.9*len(idxs))]
            trainloader = DatasetSplit(self.train_dataset, idxs_train)      
            x, y, z = torch.tensor(trainloader.x), torch.tensor(trainloader.y), torch.tensor(trainloader.sen)
            x = x.to(DEVICE)

            probas, _ = self.model(x)
            probas = probas.T[1] / torch.sum(probas, dim = 1)

            v = torch.ones(len(y)).type(torch.DoubleTensor)

            n, nz, yz, yhat = v.sum().item(), torch.zeros(self.Z).type(torch.DoubleTensor), torch.zeros(self.Z).type(torch.DoubleTensor), (probas * v).sum().item()
            z_idx = []
            for z_ in range(self.Z):
                z_idx.append(torch.where(z_ == z)[0])
                yz[z_] = (probas[z_idx[z_]] * v[z_idx[z_]]).sum().item()
                nz[z_] = v[z_idx[z_]].sum().item()

            delta = torch.zeros(self.Z)
            for z in range(self.Z):
                delta[z] = yz[z]/nz[z] - yhat/n
            mu = mu - alpha * delta

            self.model.train()
            batch_loss = []
            for _, (features, labels, sensitive) in enumerate(self.trainloader):
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

                _, logits = self.model(features)
                loss = weighted_loss(logits, labels, v)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            train_loss.append(sum(batch_loss)/len(batch_loss))

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z

            self.model.eval()

            # validation dataset inference
            acc, loss, n_yz, acc_loss, fair_loss, _ = self.inference(model = self.model) 
            list_acc.append(acc)
                
            if self.prn: 
                print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                        acc_loss, fair_loss, self.metric, self.disparity(n_yz)))

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
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd        

    def FTrain(self, num_epochs = 30, lr_g = 0.005, lr_d = 0.01, 
                 epsilon = None, num_classes = 2, ratio_gd = 3,
                 lambda_d = 0.2, init_epochs = 100):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # discriminator: estimate the sensitive attribute
        discriminator = logReg(num_classes, self.Z)
        generator = self.model

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        # Set optimizer for the local updates
        optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr_g)
        optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = lr_d)
        
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

                batch_loss.append(loss_g.item())

            loss_avg = sum(batch_loss) / len(batch_loss)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            # the number of samples which are assigned to class y and belong to the sensitive group z

            self.model.eval()
            total, correct, acc_loss, adv_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0
            n_yz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
            
            for _, (features, labels, sensitive) in enumerate(self.validloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE).type(torch.LongTensor)
                
                # Inference
                outputs, logits = self.model(features)
                _, adv_logits = discriminator(logits)

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
                    
            if self.prn: 
                print("Predictor loss: %.2f | adversary loss %.2f | %s = %.2f" % (
                        acc_loss, adv_loss, self.metric, self.disparity(n_yz)))

            train_accuracy.append(accuracy)

            # print global training loss after every 'i' rounds
            if self.prn:
                if (epoch+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {epoch+1} global rounds:')
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
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd

    # only support z == 2
    def FairBatch(self, num_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)

        # the number of samples whose label is y and sensitive attribute is z
        m_yz, lbd = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

        for y in [0,1]:
            for z in range(self.Z):
                lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

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
                
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]

                # print(logits)
                loss = weighted_loss(logits, labels, v)
                # if global_round == 1: print(loss)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss)/len(batch_loss)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz, loss_yz = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
                    loss_yz[(y,z)] = 0

            self.model.eval()
                # validation dataset inference
            acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference(model = self.model, option = 'FairBatch', validloader=self.trainloader) 
            list_acc.append(acc)
    
                
            if self.prn: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                 acc_loss, fair_loss, self.metric, self.disparity(n_yz)))
                
            # update the lambda according to the paper -> see Section A.1 of FairBatch
            # works well! The real batch size would be slightly different from the setting
            for y, z in loss_yz:
                loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

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
            print(f' \n Results after {num_epochs} global rounds of training:')
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
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0
        
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
            
            for y,z in n_yz:
                n_yz[(y,z)] += torch.sum((sensitive == z) & (pred_labels == y)).item()  

        accuracy = correct/total

        return accuracy, n_yz

    def postbc_test_inference(self, mu):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        model = self.model

        model.eval()
        total, correct = 0.0, 0.0
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0
        
        test_dataset = self.test_dataset
        testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False)

        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels =  labels.to(DEVICE).type(torch.LongTensor)
            # Inference
            outputs, _ = model(features)

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
            
            for y,z in n_yz:
                n_yz[(y,z)] += torch.sum((sensitive == z) & (pred_labels == y)).item()  

        accuracy = correct/total

        return accuracy, n_yz

    def inference(self, option = 'unconstrained', penalty = 100, model = None, validloader = None):
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
        n_yz, loss_yz = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
        
        for _, (features, labels, sensitive) in enumerate(validloader):
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
                
                if option == "FairBatch":
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx[yz]], 
                                                    labels[group_boolean_idx[yz]], 
                                         outputs[group_boolean_idx[yz]], sensitive[group_boolean_idx[yz]], 
                                         penalty)
                    loss_yz[yz] += loss_yz_
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(option, logits, 
                                                        labels, outputs, sensitive, penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if option in ["FairBatch", "FB-Variant1"]:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None
