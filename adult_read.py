import os
import pandas as pd
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

def process_adult(filename):
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'salary']

    adult = pd.read_csv(os.path.join('adult', filename), delimiter = ',', header = None)
    adult.columns =  header

    # sensitive attribute
    sen_var = ["sex"]
    # categorical attributes
    cg_var = ['workclass', 'education', 'marital-status', 'occupation',
                                    'relationship', 'race', 'native-country', 'sex'] 
    # continuous attributes
    cont_var = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
                           "hours-per-week"]
    # predict variable
    pred_var = "salary"

    # apply one-hot encoding to convert the categorical attributes into vectors
    adult = pd.get_dummies(adult, columns = cg_var)

    # normalize numerical attributes to the range within [0, 1]
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec-minimum)/(maximum-minimum)

    adult[cont_var] = adult[cont_var].apply(scale, axis = 0)
    
    # convert the pred variable into integer
    adult[pred_var] = adult[pred_var].astype('category').cat.codes
    return adult

class LoadData(Dataset):
    def __init__(self, df, pred_var):
        self.y = df[pred_var].values
        self.x = df.drop(pred_var, axis = 1).values
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.y.shape[0]

if __name__ == "adult_read":
    adult = process_adult('adult.data')
    test = process_adult('adult.test')

    np.random.seed(1)
    adult_private_idx = adult[adult['workclass_ Private'] == 1].index
    adult_others_idx = adult[adult['workclass_ Private'] == 0].index

    client1_idx = np.concatenate((np.random.choice(adult_private_idx, int(.8*len(adult_private_idx)), replace = False),
                                    np.random.choice(adult_others_idx, int(.2*len(adult_others_idx)), replace = False)))
    client2_idx = np.array(list(set(adult.index) - set(client1_idx)))

    client1 = adult.iloc[client1_idx]
    client1 = client1.reset_index(drop=True)

    client2 = adult.iloc[client2_idx]
    client2 = client2.reset_index(drop=True)

    clients = [client1, client2]

    ### torch preparation ###
    ################## MODEL SETTING ########################
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    RANDOM_SEED = 123
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    client1_dataset = LoadData(client1, 'salary')
    client1_loader = DataLoader(dataset = client1_dataset,
                                batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS)

    client2_dataset = LoadData(client2, 'salary')
    client2_loader = DataLoader(dataset = client2_dataset,
                                batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS)

    test_dataset = LoadData(test, 'salary')
    test_loader = DataLoader(dataset = test_dataset,
                                batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS)