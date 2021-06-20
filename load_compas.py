import numpy as np

from utils import *
import torch
   
if __name__ == "load_compas":
    sensitive_attributes = ['sex', 'race']
    categorical_attributes = ['age_cat', 'c_charge_degree', 'c_charge_desc']
    continuous_attributes = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
            'priors_count', 'c_charge_degree', 'c_charge_desc','two_year_recid']
    label_name = 'two_year_recid'

    compas = process_csv('compas', 'compas-scores-two-years.csv', label_name, 0, sensitive_attributes, ['Female', 'African-American'], categorical_attributes, continuous_attributes, features_to_keep)
    train = compas.iloc[:int(len(compas)*.9)]
    test = compas.iloc[int(len(compas)*.9):]

    np.random.seed(1)
    torch.manual_seed(0)
    client1_idx = train[train.age > 0.1].index 
    client2_idx = train[train.age <= 0.1].index
    compas_mean_sensitive = train['z'].mean()
    compas_z = len(set(compas.z))

    clients_idx = [client1_idx, client2_idx]

    compas_num_features = len(compas.columns) - 1
    compas_train = LoadData(train, label_name, 'z')
    compas_test = LoadData(test, label_name, 'z')
    
    compas_info = [compas_train, compas_test, clients_idx]