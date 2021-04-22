import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader, Dataset
import torch
from utils import *

########################
####### setting ########
########################
X_DIST = {0:{"mean":(-2,-2), "cov":np.array([[10,1], [1,3]])}, 
     1:{"mean":(2,2), "cov":np.array([[5,1], [1,5]])}}

def X_PRIME(x):
    return (x[0]*np.cos(np.pi/4) - x[1]*np.sin(np.pi/4), 
            x[0]*np.sin(np.pi/4) + x[1]*np.cos(np.pi/4))
def Z_MEAN(x, y):
    """
    Given x and y, the probability of z = 1.
    """
    x_transform = X_PRIME(x)
    return multivariate_normal.pdf(x_transform, mean = X_DIST[1]["mean"], cov = X_DIST[1]["cov"])/(
        multivariate_normal.pdf(x_transform, mean = X_DIST[1]["mean"], cov = X_DIST[1]["cov"]) + 
        multivariate_normal.pdf(x_transform, mean = X_DIST[0]["mean"], cov = X_DIST[0]["cov"])
    )
########################

def dataGenerate(seed = 432, num_samples = 3500, train_samples = 3000, 
                y_mean = 0.6):
    np.random.seed(seed)
        
    ys = np.random.binomial(n = 1, p = y_mean, size = num_samples)

    xs, zs = [], []

    for y in ys:
        x = np.random.multivariate_normal(mean = X_DIST[y]["mean"], cov = X_DIST[y]["cov"], size = 1)[0]
        z = np.random.binomial(n = 1, p = Z_MEAN(x,y), size = 1)[0]
        xs.append(x)
        zs.append(z)

    data = pd.DataFrame(zip(np.array(xs).T[0], np.array(xs).T[1], ys, zs), columns = ["x1", "x2", "y", "z"])
    train_data = data[:train_samples]
    test_data = data[train_samples:]
    
    z1_idx = train_data[train_data.z == 1].index
    z0_idx = train_data[train_data.z == 0].index
    
    # 3 clients: 
    #           client 1: %60 z = 1, %10 z = 0
    #           client 2: %30 z = 1, %50 z = 0
    #           client 3: %10 z = 1, %40 z = 0

    client1_idx = np.concatenate((z1_idx[:int(.6*len(z1_idx))], z0_idx[:int(.1*len(z0_idx))]))
    client2_idx = np.concatenate((z1_idx[int(.6*len(z1_idx)):int(.9*len(z1_idx))], z0_idx[int(.1*len(z0_idx)):int(.6*len(z0_idx))]))
    client3_idx = np.concatenate((z1_idx[int(.9*len(z1_idx)):], z0_idx[int(.6*len(z0_idx)):]))

    clients_idx = [client1_idx, client2_idx, client3_idx]
    train_dataset = LoadData(train_data, "y", "z")
    test_dataset = LoadData(test_data, "y", "z")

    synthetic_info = [train_dataset, test_dataset, clients_idx]
    return synthetic_info