# simulations for demographic parity
# import the training method 
from Server import *

import time

def runSim(num_sim = 20, train_samples = 3000, test_samples = 100, learning_rate = 0.005, num_rounds = 5, 
          local_epochs = 40, alpha = 1, metric = "Demographic disparity", adaptive_alpha = True, option = "FairBatch",
          optimizer = 'adam', penalty = 500, adjusting_rounds = 10, adjusting_epochs = 30, 
          adjusting_alpha = 0.7, epsilon = 0.02, adaptive_lr = True, test_lr = 0.01, test_rounds = 3,
          lr_g = 0.005, lr_d = 0.01, init_epochs = 50, lambda_d = 0.8, bs_iter = 10, alpha_decay = 0.5):
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
        
        server = Server(logReg(num_features=3, num_classes=2, seed=seed), synthetic_info, seed = seed, ret = True, train_prn = False, metric = metric)
        # train the model with the synthetic dataset
        if option == 'unconstrained':
            test_acc_i, rd_i = server.Unconstrained(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer)

        elif option == 'local zafar':
            test_acc_i, rd_i = server.LocalZafar(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, penalty = penalty, epsilon = epsilon)

        elif option == 'threshold adjusting':
            server.Unconstrained(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer)
            test_acc_i, rd_i = server.ThresholdAdjust(num_rounds = adjusting_rounds, local_epochs = adjusting_epochs, learning_rate = adjusting_alpha, 
                 epsilon = epsilon)

        elif option == 'FairBatch':
            test_acc_i, rd_i = server.FairBatch(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, adaptive_alpha = adaptive_alpha, alpha = alpha)

        elif option == 'adversarial learning':
            server.Unconstrained(num_rounds = 1, local_epochs = local_epochs, learning_rate = 0.01, optimizer = 'adam')
            test_acc_i, rd_i = server.AdversarialLearning(num_rounds = num_rounds-1, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, epsilon = epsilon, alpha = alpha, adaptive_lr = adaptive_lr)

        elif option == 'bias correcting':
            test_acc_i, rd_i = server.BiasCorrecting(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, epsilon = epsilon, alpha = alpha)

        elif option == 'zafar':
            test_acc_i, rd_i = server.Zafar(test_rounds = test_rounds, test_lr = test_lr, num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, penalty = penalty)

        elif option == 'ftrain':
            test_acc_i, rd_i = server.FTrain(num_rounds = num_rounds, local_epochs = local_epochs, init_epochs = init_epochs, 
                lr_g = lr_g, lr_d = lr_d, lambda_d = lambda_d)

        elif option == 'bc-variant1':
            test_acc_i, rd_i = server.BCVariant1(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, epsilon = epsilon, alpha = alpha)
        
        elif option == 'bc-variant2':
            test_acc_i, rd_i = server.BCVariant2(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, epsilon = epsilon, bs_iter = bs_iter)

        elif option == 'bc-variant3':
            test_acc_i, rd_i = server.BCVariant3(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, epsilon = epsilon, alpha = alpha)
                
        elif option == 'fb-variant1':
            test_acc_i, rd_i = server.FBVariant2(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, alpha = alpha)

        elif option == 'fb-variant2':
            test_acc_i, rd_i = server.FBVariant2(num_rounds = num_rounds, local_epochs = local_epochs, learning_rate = learning_rate, 
                optimizer = optimizer, alpha = alpha, alpha_decay = alpha_decay)

        else:
            print('Approach %s is not supported!' % option)
            return None

        test_acc.append(test_acc_i)
        rd.append(rd_i)
        
        print("      Accuracy: %.2f%%  %s: %.2f" % (test_acc_i * 100, metric, rd_i))
        
    mean_acc, std_acc, mean_rd, std_rd = np.mean(test_acc), np.std(test_acc), np.mean(rd), np.std(rd)
    print("| Test Accuracy: %.3f(%.3f) | %s: %.3f(%.3f) |" % (mean_acc, std_acc, metric, mean_rd, std_rd))
    
    print("| Time elapsed: %.2f seconds |" % (time.time() - start))
    return mean_acc, std_acc, mean_rd, std_rd