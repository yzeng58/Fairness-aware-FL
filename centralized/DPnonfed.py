from DPFair import *

def runNonFed(data_info, method = 'FairBatch', learning_rate = 0.005, num_epochs = 300, alpha = 1, num_features = 3,
                optimizer = 'adam', seed = 123, penalty = (100,100,100), Z = 2, lr_g = 0.005, lr_d = 0.01, 
                  ratio_gd = 3,
                 lambda_d = (0.9, 0.8, 0.8), init_epochs = 100):
     
    acc, fair, n = 0, 0, 0
    clients_idx = data_info[2]
    for c in range(len(clients_idx)):
        client_info = copy.deepcopy(data_info)
        client_info[0] = DatasetSplit(client_info[0], client_info[2][c])
        server = Server(logReg(num_features=num_features, num_classes=2, seed=seed), client_info, seed = seed, ret = True)

        if method == 'FairBatch':
            acc_, fair_ = server.FairBatch(num_epochs = num_epochs, learning_rate = learning_rate, optimizer = optimizer,
                            alpha = alpha)
        elif method == 'FC': 
            acc_, fair_ = server.FairConstraints(num_epochs = num_epochs, learning_rate = learning_rate, penalty = penalty[c],
                    optimizer = optimizer)
        elif method == 'FTrain':
            acc_, fair_ = server.FTrain(num_epochs = num_epochs, lr_g = lr_g, lr_d = lr_d, num_classes = Z, ratio_gd = ratio_gd, lambda_d = lambda_d[c], init_epochs = init_epochs)
        n += len(clients_idx[c])
        acc += acc_ * len(clients_idx[c])
        fair += fair_ * len(clients_idx[c])
    return acc/n, fair/n
