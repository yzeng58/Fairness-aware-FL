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
    def __init__(self, df, pred_var, sen_var):
        self.y = df[pred_var].values
        self.x = df.drop(pred_var, axis = 1).values
        self.sen = df[sen_var].values
    
    def __getitem__(self, index):
        return torch.tensor(self.x[index]), torch.tensor(self.y[index]), torch.tensor(self.sen[index])
    
    def __len__(self):
        return self.y.shape[0]
   
if __name__ == "utils":
    adult = process_adult('adult.data')
    test = process_adult('adult.test')
    test['native-country_ Holand-Netherlands'] = 0
    test = test[adult.columns]

    np.random.seed(1)
    adult_private_idx = adult[adult['workclass_ Private'] == 1].index
    adult_others_idx = adult[adult['workclass_ Private'] == 0].index

    client1_idx = np.concatenate((np.random.choice(adult_private_idx, int(.8*len(adult_private_idx)), replace = False),
                                    np.random.choice(adult_others_idx, int(.2*len(adult_others_idx)), replace = False)))
    client2_idx = np.array(list(set(adult.index) - set(client1_idx)))
    clients_idx = [client1_idx, client2_idx]
    client1 = adult.iloc[clients_idx[0]]
    client1 = client1.reset_index(drop=True)

    client2 = adult.iloc[clients_idx[1]]
    client2 = client2.reset_index(drop=True)

    clients = [client1, client2]

    ################## torch preparation ##################

    ################## MODEL SETTING ########################
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    RANDOM_SEED = 123
    NUM_EPOCHS = 10 
    LEARNING_RATE = 0.05
    FRAC = 1 # the fraction of clients
    NUM_CLIENTS = 2
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    OPTIMIZER = 'adam'
    LOCAL_EPOCHS = 10
    NUM_FEATURES = len(adult.columns)-1
    SEN_VAR = "sex"
    #########################################################
    client1_dataset = LoadData(client1, 'salary', 'sex_ Female')
    client1_loader = DataLoader(dataset = client1_dataset,
                                batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS)

    client2_dataset = LoadData(client2, 'salary', 'sex_ Female')
    client2_loader = DataLoader(dataset = client2_dataset,
                                batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS)

    test_dataset = LoadData(test, 'salary', 'sex_ Female')
    test_loader = DataLoader(dataset = test_dataset,
                                batch_size = BATCH_SIZE,
                                num_workers = NUM_WORKERS)
    
    train_dataset = LoadData(adult, 'salary', 'sex_ Female')
    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS)

    torch.manual_seed(0)

    # test whether dataset can be loaded correctly
    num_epochs = 2
    for epoch in range(num_epochs):

        for batch_idx, (x, y, sen) in enumerate(client1_loader):
            
            print('Epoch:', epoch+1, end='')
            print(' | Batch index:', batch_idx, end='')
            print(' | Batch size:', y.size()[0])
            
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            sen = sen.to(DEVICE)
            
            print('break minibatch for-loop')
            break