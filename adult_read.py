import os
import pandas as pd
import numpy as np

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

adult = process_adult('adult.data')
test = process_adult('adult.test')

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