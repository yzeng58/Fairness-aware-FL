######################################################
### Pre-processing code (leave here for reference) ###
######################################################
# import pandas as pd
# import numpy as np
# import os
# from utils import LoadData

# df = pd.read_csv(os.path.join('bank', 'bank-full.csv'), sep = ';')
# q1 = df.age.quantile(q = 0.2)
# q1_idx = np.where(df.age <= q1)[0]
# q2 = df.age.quantile(q = 0.4)
# q2_idx = np.where((q1 < df.age) & (df.age <= q2))[0]
# q3 = df.age.quantile(q = 0.6)
# q3_idx = np.where((q2 < df.age) & (df.age <= q3))[0]
# q4 = df.age.quantile(q = 0.8)
# q4_idx = np.where((q3 < df.age) & (df.age <= q4))[0]
# q5_idx = np.where(df.age > q4)[0]
# df.loc[q1_idx, 'age'] = 0
# df.loc[q2_idx, 'age'] = 1
# df.loc[q3_idx, 'age'] = 2
# df.loc[q4_idx, 'age'] = 3
# df.loc[q5_idx, 'age'] = 4
# df.to_csv(os.path.join('bank', 'bank_cat_age.csv'))
######################################################

import numpy as np
from utils import *
import torch

if __name__ == "load_bank":
    np.random.seed(1)
    torch.manual_seed(0)
    sensitive_attributes = ['age']
    categorical_attributes = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    continuous_attributes = ['balance', 'duration', 'campaign', 'pdays', 'previous']
    features_to_keep = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 
                        'balance', 'duration', 'campaign', 'pdays', 'previous', 'y']
    label_name = 'y'

    bank = process_csv('bank', 'bank_cat_age.csv', label_name, 'yes', sensitive_attributes, None, categorical_attributes, continuous_attributes, features_to_keep, na_values = [])
    bank = bank.sample(frac=1).reset_index(drop=True)

    train = bank.iloc[:int(len(bank)*.9)]
    test = bank.iloc[int(len(bank)*.9):]

    loan_idx = np.where(train.loan_no == 1)[0]
    loan_no_idx = np.where(train.loan_no == 0)[0]
    client1_idx = np.concatenate((loan_idx[:int(len(loan_idx)*.5)], loan_no_idx[:int(len(loan_no_idx)*.2)]))
    client2_idx = np.concatenate((loan_idx[int(len(loan_idx)*.5):int(len(loan_idx)*.6)], loan_no_idx[int(len(loan_no_idx)*.2):int(len(loan_no_idx)*.8)]))
    client3_idx = np.concatenate((loan_idx[int(len(loan_idx)*.6):], loan_no_idx[int(len(loan_no_idx)*.8):]))
    np.random.shuffle(client1_idx)
    np.random.shuffle(client2_idx)
    np.random.shuffle(client3_idx)

    bank_mean_sensitive = train['z'].mean()
    bank_z = len(set(bank.z))

    clients_idx = [client1_idx, client2_idx, client3_idx]

    bank_num_features = len(bank.columns) - 1
    bank_train = LoadData(train, label_name, 'z')
    bank_test = LoadData(test, label_name, 'z')
    
    bank_info = [bank_train, bank_test, clients_idx]