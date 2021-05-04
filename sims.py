# import the training method 
from FairFedAvg import *

import time

def runSim(num_sim = 20, train_samples = 3000, test_samples = 100, learning_rate = 0.005, num_rounds = 5, 
          local_epochs = 40, alpha = 1, metric = "Demographic disparity", adaptive_alpha = True, option = "FairBatch",
          optimizer = 'sgd', penalty = 500, adjusting_rounds = 10, adjusting_epochs = 30, 
          adjusting_alpha = 0.7, epsilon = 0.02):
    """
    Run simulations.
    """
    
    test_acc, rd = [], []
    start = time.time()
    
    for i in range(num_sim):
        seed = int(time.time()%1000)
        print("|  Simulation : %d | " % (i+1))
        print("      seed : %d -----" % seed)
        
        # generate the synthetic dataset
        synthetic_info = dataGenerate(seed = seed, train_samples = train_samples, test_samples = test_samples)
        
        # train the model with the synthetic dataset
        test_acc_i, rd_i = train(logReg(num_features=3, num_classes=2), synthetic_info,
          option = option, optimizer = optimizer, learning_rate = learning_rate, metric = metric,
          num_rounds = num_rounds, local_epochs = local_epochs, alpha = alpha, ret = True, seed = seed,
          train_prn = False, adaptive_alpha = adaptive_alpha, penalty = penalty, 
          adjusting_alpha = adjusting_alpha, adjusting_epochs = adjusting_epochs, 
          adjusting_rounds = adjusting_rounds, epsilon = epsilon)
        
        test_acc.append(test_acc_i)
        rd.append(rd_i)
        
        print("      Accuracy: %.2f%%  %s: %.2f" % (test_acc_i * 100, metric, rd_i))
        
    mean_acc, std_acc, mean_rd, std_rd = np.mean(test_acc), np.std(test_acc), np.mean(rd), np.std(rd)
    print("| Test Accuracy: %.3f(%.3f) | %s: %.3f(%.3f) |" % (mean_acc, std_acc, metric, mean_rd, std_rd))
    
    print("| Time elapsed: %.2f seconds |" % (time.time() - start))
    return mean_acc, std_acc, mean_rd, std_rd