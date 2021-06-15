import numpy as np

from utils import *
import torch
   
if __name__ == "load_adult":
    sensitive_attributes = ['sex']
    categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 
                'native-country', 'salary']
    label_name = 'salary'

    adult = process_csv('adult', 'adult.data', label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
    test = process_csv('adult', 'adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
    test['native-country_ Holand-Netherlands'] = 0
    test = test[adult.columns]

    np.random.seed(1)
    adult_private_idx = adult[adult['workclass_ Private'] == 1].index
    adult_others_idx = adult[adult['workclass_ Private'] == 0].index
    adult_mean_sensitive = adult['z'].mean()

    client1_idx = np.concatenate((np.random.choice(adult_private_idx, int(.8*len(adult_private_idx)), replace = False),
                                    np.random.choice(adult_others_idx, int(.2*len(adult_others_idx)), replace = False)))
    client2_idx = np.array(list(set(adult.index) - set(client1_idx)))
    adult_clients_idx = [client1_idx, client2_idx]

    ################## torch preparation ##################
    adult_num_features = len(adult.columns)-1

    adult_test = LoadData(test, 'salary', 'z')

    adult_train = LoadData(adult, 'salary', 'z')

    torch.manual_seed(0)

    adult_info = [adult_train, adult_test, adult_clients_idx]
