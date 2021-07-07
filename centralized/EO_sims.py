# simulations for demographic parity
# import the training method 
from EOFair import *

import time

def runSim(num_sim = 20, train_samples = 3000, test_samples = 100, learning_rate = 0.005, num_epochs = 300, alpha = 1, option = "FairBatch",
          optimizer = 'adam', fixed_dataset = None, trace = False):
    """
    Run simulations.
    """
    
    test_acc, rd = [], []
    start = time.time()
    if fixed_dataset: synthetic_info = dataGenerate(seed = fixed_dataset, train_samples = train_samples, test_samples = test_samples)
    for i in range(num_sim):
        seed = int(time.time()%1000)
        print("|  Simulation : %d | " % (i+1))
        print("      seed : %d -----" % seed)
        
        # generate the synthetic dataset
        if fixed_dataset == None: 
            synthetic_info = dataGenerate(seed = seed, train_samples = train_samples, test_samples = test_samples)
        
        server = Server(logReg(num_features=3, num_classes=2, seed=seed), synthetic_info, seed = seed, ret = True, train_prn = False)
        # train the model with the synthetic dataset
        if option == 'FairBatch':
            test_acc_i, rd_i = server.FairBatch(num_epochs = num_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, alpha = alpha, trace = trace)

        else:
            print('Approach %s is not supported!' % option)
            return None

        test_acc.append(test_acc_i)
        rd.append(rd_i)
        
        if not trace:
            print("      Accuracy: %.2f%%  %s: %.2f" % (test_acc_i * 100, metric, rd_i))
        else:
            print(test_acc_i)
            print(rd_i)
        
    if trace:
        test_acc, rd = np.array(test_acc), np.array(rd)
        mean_acc, mean_rd = test_acc.mean(axis = 0), rd.mean(axis = 0)
        return mean_acc.tolist(), mean_rd.tolist()
    else:
        mean_acc, std_acc, mean_rd, std_rd = np.mean(test_acc), np.std(test_acc), np.mean(rd), np.std(rd)
        print("| Test Accuracy: %.3f(%.3f) | %s: %.3f(%.3f) |" % (mean_acc, std_acc, metric, mean_rd, std_rd))
        print("| Time elapsed: %.2f seconds |" % (time.time() - start))
        return mean_acc, std_acc, mean_rd, std_rd