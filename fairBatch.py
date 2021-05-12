import numpy as np
import random, itertools
from torch.utils.data.sampler import Sampler


class FairBatch(Sampler):
    """FairBatch (Sampler in DataLoader).
    
    This class is for implementing batch selection of FairBatch.
        
    """
    def __init__(self, train_dataset, lbd, client_idx, batch_size, replacement = False, seed = 0):
        """Initializes FairBatch."""

        self.batch_size = batch_size
        self.N = train_dataset.y.shape[0]
        self.batch_num = int(self.N / self.batch_size)
        self.lbd = lbd
        self.seed = seed
        
        self.yz_index, self.yz_size = {}, {}
        
        for y, z in itertools.product([0,1], [0,1]):
                self.yz_index[(y,z)] = np.where((train_dataset.y == y) & (train_dataset.sen == z))[0]
                self.yz_size[(y,z)] = len(self.yz_index[(y,z)]) / self.N * self.batch_size
        

    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement = False):
        """Selects a certain number of batches based on the given batch size.
        
        Args: 
            batch_size: An integer for the data size in a batch.
            full_index: An array containing the candidate data indices.
            batch_num: An integer indicating the number of batches.
            replacement: A boolean indicating whether a batch consists of data with or without replacement.
        
        Returns:
            Indices that indicate the data.
            
        """
        np.random.seed(self.seed)
        random.seed(self.seed)

        select_index = []
        
        if replacement == True:
            for _ in range(batch_num):
                select_index.append(np.random.choice(full_index, batch_size, replace = False))
        else:
            tmp_index = full_index.copy()
            random.shuffle(tmp_index)
            
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(full_index):
                    select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                    start_idx = len(full_index)-start_idx
                else:
                    select_index.append(tmp_index[start_idx:start_idx + batch_size])
                    start_idx += batch_size
        return select_index

    
    def __iter__(self):
        """Iters the full process of FairBatch for serving the batches to training.
        
        Returns:
            Indices that indicate the data in each batch.
            
        """
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Get the indices for each class
        sort_index_y_1_z_1 = self.select_batch_replacement(int(self.lbd[(1,1)] * (self.yz_size[(1,1)] + self.yz_size[(1,0)])), self.yz_index[(1,1)], self.batch_num)
        sort_index_y_0_z_1 = self.select_batch_replacement(int(self.lbd[(0,1)] * (self.yz_size[(0,1)] + self.yz_size[(0,0)])), self.yz_index[(0,1)], self.batch_num)
        sort_index_y_1_z_0 = self.select_batch_replacement(int(self.lbd[(1,0)] * (self.yz_size[(1,1)] + self.yz_size[(1,0)])), self.yz_index[(1,0)], self.batch_num)
        sort_index_y_0_z_0 = self.select_batch_replacement(int(self.lbd[(0,0)] * (self.yz_size[(0,1)] + self.yz_size[(0,0)])), self.yz_index[(0,0)], self.batch_num)
            

        for i in range(self.batch_num):
            key_in_fairbatch = sort_index_y_0_z_0[i].copy()
            key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_0[i].copy()))
            key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_0_z_1[i].copy()))
            key_in_fairbatch = np.hstack((key_in_fairbatch, sort_index_y_1_z_1[i].copy()))

            random.shuffle(key_in_fairbatch)
            for idx in key_in_fairbatch:
                yield idx
            # yield key_in_fairbatch
                               

    def __len__(self):
        """Returns the length of data."""
        
        return self.N