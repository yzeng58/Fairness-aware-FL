from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from scipy.stats import multivariate_normal
import torch, random, copy, os

class LoadData(Dataset):
    def __init__(self, df, pred_var, sen_var):
        self.y = df[pred_var].values
        self.x = df.drop(pred_var, axis = 1).values
        self.sen = df[sen_var].values
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index]), torch.tensor(self.sen[index])
    
    def __len__(self):
        return self.y.shape[0]

class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.x = self.dataset.x[self.idxs]
        self.y = self.dataset.y[self.idxs]
        self.sen = self.dataset.sen[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        feature, label, sensitive = self.dataset[self.idxs[item]]
        return feature, label, sensitive
        # return self.x[item], self.y[item], self.sen[item]
    
class logReg(torch.nn.Module):
    """
    Logistic regression model.
    """
    def __init__(self, num_features, num_classes, seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x.float())
        probas = torch.sigmoid(logits)
        return probas.type(torch.FloatTensor), logits

def logit_compute(probas):
    return torch.log(probas/(1-probas))
    
def riskDifference(n_yz, absolute = True):
    """
    Given a dictionary of number of samples in different groups, compute the risk difference.
    |P(Group1, pos) - P(Group2, pos)| = |N(Group1, pos)/N(Group1) - N(Group2, pos)/N(Group2)|
    """
    n_z1 = max(n_yz[(1,1)] + n_yz[(0,1)], 1)
    n_z0 = max(n_yz[(0,0)] + n_yz[(1,0)], 1)
    if absolute:
        return abs(n_yz[(1,1)]/n_z1 - n_yz[(1,0)]/n_z0)
    else:
        return n_yz[(1,1)]/n_z1 - n_yz[(1,0)]/n_z0

def pRule(n_yz):
    """
    Compute the p rule level.
    min(P(Group1, pos)/P(Group2, pos), P(Group2, pos)/P(Group1, pos))
    """
    return min(n_yz[(1,1)]/n_yz[(1,0)], n_yz[(1,0)]/n_yz[(1,1)])

def DPDisparity(n_yz, each_z = False):
    """
    Same metric as FairBatch. Compute the demographic disparity.
    max(|P(pos | Group1) - P(pos)|, |P(pos | Group2) - P(pos)|)
    """
    z_set = sorted(list(set([z for _, z in n_yz.keys()])))
    p_y1_n, p_y1_d, n_z = 0, 0, []
    for z in z_set:
        p_y1_n += n_yz[(1,z)]
        n_z.append(max(n_yz[(1,z)] + n_yz[(0,z)], 1))
        for y in [0,1]:
            p_y1_d += n_yz[(y,z)]
    p_y1 = p_y1_n / p_y1_d

    if not each_z:
        return max([abs(n_yz[(1,z)]/n_z[z] - p_y1) for z in z_set])
    else:
        return [n_yz[(1,z)]/n_z[z] - p_y1 for z in z_set]

def EODisparity(n_eyz, each_z = False):
    """
    Equal opportunity disparity: max_z{|P(yhat=1|z=z,y=1)-P(yhat=1|y=1)|}

    Parameter:
    n_eyz: dictionary. #(yhat=e,y=y,z=z)
    """
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    if not each_z:
        eod = 0
        p11 = sum([n_eyz[(1,1,z)] for z in z_set]) / sum([n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
        for z in z_set:
            eod_z = abs(n_eyz[(1,1,z)]/(n_eyz[(0,1,z)] + n_eyz[(1,1,z)]) - p11)
            if eod < eod_z:
                eod = eod_z
        return eod
    else:
        eod = []
        p11 = sum([n_eyz[(1,1,z)] for z in z_set]) / sum([n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
        for z in z_set:
            eod_z = n_eyz[(1,1,z)]/(n_eyz[(0,1,z)] + n_eyz[(1,1,z)]) - p11
            eod.append(eod_z)
        return eod

def average_weights(w, clients_idx, idx_users):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    num_samples = 0
    for i in range(1, len(w)):
        num_samples += len(clients_idx[idx_users[i]])
        for key in w_avg.keys():            
            w_avg[key] += w[i][key] * len(clients_idx[idx_users[i]])
        
    for key in w_avg.keys(): 
        w_avg[key] = torch.div(w_avg[key], num_samples)
    return w_avg

def weighted_average_weights(w, nc, n):
    w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        for key in w_avg.keys():            
            w_avg[key] += w[i][key] * nc[i]
        
    for key in w_avg.keys(): 
        w_avg[key] = torch.div(w_avg[key], n)
    return w_avg

def loss_func(option, logits, targets, outputs, sensitive, larg = 1):
    """
    Loss function. 
    """

    acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
    fair_loss0 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits.T[0] - torch.mean(logits.T[0]))
    fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0)) 
    fair_loss1 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits.T[1] - torch.mean(logits.T[1]))
    fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1)) 
    fair_loss = fair_loss0 + fair_loss1

    if option == 'local zafar':
        return acc_loss + larg*fair_loss, acc_loss, larg*fair_loss
    elif option == 'FB_inference':
        # acc_loss = torch.sum(torch.nn.BCELoss(reduction = 'none')((outputs.T[1]+1)/2, torch.ones(logits.shape[0])))
        acc_loss = F.cross_entropy(logits, torch.ones(logits.shape[0]).type(torch.LongTensor), reduction = 'sum')
        return acc_loss, acc_loss, fair_loss
    else:
        return acc_loss, acc_loss, larg*fair_loss

def eo_loss(logits, targets, sensitive, larg, mean_z1 = None, left = None, option = 'local fc'):
    acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
    y1_idx = torch.where(targets == 1)
    if option == 'unconstrained':
        return acc_loss
    if left:
        fair_loss = torch.mean(torch.mul(sensitive[y1_idx] - mean_z1, logits.T[0][y1_idx] - torch.mean(logits.T[0][y1_idx])))
        return acc_loss - larg * fair_loss
    elif left == False: 
        fair_loss = torch.mean(torch.mul(sensitive[y1_idx] - mean_z1, logits.T[0][y1_idx] - torch.mean(logits.T[0][y1_idx])))
        return acc_loss + larg * fair_loss
    else:
        fair_loss0 = torch.mul(sensitive[y1_idx] - sensitive.type(torch.FloatTensor).mean(), logits.T[0][y1_idx] - torch.mean(logits.T[0][y1_idx]))
        fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0)) 
        fair_loss1 = torch.mul(sensitive[y1_idx] - sensitive.type(torch.FloatTensor).mean(), logits.T[1][y1_idx] - torch.mean(logits.T[1][y1_idx]))
        fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1)) 
        fair_loss = fair_loss0 + fair_loss1
        return acc_loss + larg * fair_loss

def zafar_loss(logits, targets, outputs, sensitive, larg, mean_z, left):
    acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
    fair_loss =  torch.mean(torch.mul(sensitive - mean_z, logits.T[0] - torch.mean(logits.T[0])))
    if left:
        return acc_loss - larg * fair_loss
    else:
        return acc_loss + larg * fair_loss

def weighted_loss(logits, targets, weights):
    acc_loss = F.cross_entropy(logits, targets, reduction = 'none')
    weights_sum = weights.sum().item()
    acc_loss = torch.sum(acc_loss * weights / weights_sum)
    return acc_loss
    
def al_loss(logits, targets, adv_logits, adv_targets):
    acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
    adv_loss = F.cross_entropy(adv_logits, adv_targets)
    return acc_loss, adv_loss

## Synthetic data generation ##
########################
####### setting ########
########################
X_DIST = {0:{"mean":(-2,-2), "cov":np.array([[10,1], [1,3]])}, 
     1:{"mean":(2,2), "cov":np.array([[5,1], [1,5]])}}

def X_PRIME(x):
    return (x[0]*np.cos(np.pi/4) - x[1]*np.sin(np.pi/4), 
            x[0]*np.sin(np.pi/4) + x[1]*np.cos(np.pi/4))
def Z_MEAN(x):
    """
    Given x, the probability of z = 1.
    """
    x_transform = X_PRIME(x)
    return multivariate_normal.pdf(x_transform, mean = X_DIST[1]["mean"], cov = X_DIST[1]["cov"])/(
        multivariate_normal.pdf(x_transform, mean = X_DIST[1]["mean"], cov = X_DIST[1]["cov"]) + 
        multivariate_normal.pdf(x_transform, mean = X_DIST[0]["mean"], cov = X_DIST[0]["cov"])
    )
########################

# 3 clients: 
#           client 1: %50 z = 1, %20 z = 0
#           client 2: %30 z = 1, %40 z = 0
#           client 3: %20 z = 1, %40 z = 0

def dataGenerate(seed = 432, train_samples = 3000, test_samples = 500, 
                y_mean = 0.6, client_split = ((.5, .2), (.3, .4), (.2, .4))):
    """
    Generate dataset consisting of two sensitive groups.
    """
    np.random.seed(seed)
    random.seed(seed)
        
    num_samples = train_samples + test_samples
    ys = np.random.binomial(n = 1, p = y_mean, size = num_samples)

    xs, zs = [], []

    for y in ys:
        x = np.random.multivariate_normal(mean = X_DIST[y]["mean"], cov = X_DIST[y]["cov"], size = 1)[0]
        z = np.random.binomial(n = 1, p = Z_MEAN(x), size = 1)[0]
        xs.append(x)
        zs.append(z)

    data = pd.DataFrame(zip(np.array(xs).T[0], np.array(xs).T[1], ys, zs), columns = ["x1", "x2", "y", "z"])
    train_data = data[:train_samples]
    test_data = data[train_samples:]
    
    z1_idx = train_data[train_data.z == 1].index
    z0_idx = train_data[train_data.z == 0].index

    client1_idx = np.concatenate((z1_idx[:int(client_split[0][0]*len(z1_idx))], z0_idx[:int(client_split[0][1]*len(z0_idx))]))
    client2_idx = np.concatenate((z1_idx[int(client_split[0][0]*len(z1_idx)):int((client_split[0][0] + client_split[1][0])*len(z1_idx))],
                                  z0_idx[int(client_split[0][1]*len(z0_idx)):int((client_split[0][1] + client_split[1][1])*len(z0_idx))]))
    client3_idx = np.concatenate((z1_idx[int((client_split[0][0] + client_split[1][0])*len(z1_idx)):], z0_idx[int((client_split[0][1] + client_split[1][1])*len(z0_idx)):]))
    random.shuffle(client1_idx)
    random.shuffle(client2_idx)
    random.shuffle(client3_idx)

    clients_idx = [client1_idx, client2_idx, client3_idx]
    train_dataset = LoadData(train_data, "y", "z")
    test_dataset = LoadData(test_data, "y", "z")

    synthetic_info = [train_dataset, test_dataset, clients_idx]
    return synthetic_info

def process_csv(dir_name, filename, label_name, favorable_class, sensitive_attributes, privileged_classes, categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = 'infer', columns = None):
    """
    process the adult file: scale, one-hot encode
    only support binary sensitive attributes -> [gender, race] -> 4 sensitive groups 
    """

    df = pd.read_csv(os.path.join('..', dir_name, filename), delimiter = ',', header = header, na_values = na_values)
    if header == None: df.columns = columns
    df = df[features_to_keep]

    # apply one-hot encoding to convert the categorical attributes into vectors
    df = pd.get_dummies(df, columns = categorical_attributes)

    # normalize numerical attributes to the range within [0, 1]
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec-minimum)/(maximum-minimum)
    
    df[continuous_attributes] = df[continuous_attributes].apply(scale, axis = 0)
    df.loc[df[label_name] != favorable_class, label_name] = 'SwapSwapSwap'
    df.loc[df[label_name] == favorable_class, label_name] = 1
    df.loc[df[label_name] == 'SwapSwapSwap', label_name] = 0
    df[label_name] = df[label_name].astype('category').cat.codes
    if len(sensitive_attributes) > 1:
        if privileged_classes != None:
            for i in range(len(sensitive_attributes)):
                df.loc[df[sensitive_attributes[i]] != privileged_classes[i], sensitive_attributes[i]] = 0
                df.loc[df[sensitive_attributes[i]] == privileged_classes[i], sensitive_attributes[i]] = 1
        df['z'] = list(zip(*[df[c] for c in sensitive_attributes]))
        df['z'] = df['z'].astype('category').cat.codes
    else:
        df['z'] = df[sensitive_attributes[0]].astype('category').cat.codes
    df = df.drop(columns = sensitive_attributes)
    return df
