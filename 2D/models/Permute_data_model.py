import numpy as np
import torch
import torch.nn as nn
class Permute_data(nn.Module):
    '''
    Args: 
    x: input (BXCXHXW)
    To permute the data channel-wise. This operation called during both the training and testing.
    '''
    def __init__(self, input_data, seed):
        super(Permute_data,self).__init__()
        #fixed seed
        np.random.seed(seed)
        self.Permute_data = np.random.permutation(input_data)
        np.random.seed()
        Permute_sample = np.zeros((self.Permute_data.shape))
        for i, j in enumerate(self.Permute_data):
            Permute_sample[j] = i
        self.Permute_sample = Permute_sample
    def forward(self, x, sample_the_data=False):
        if sample_the_data == False:
            y = x[:, self.Permute_data]
            return y
        else:
            y1 = x[:, self.Permute_sample]
            return y1
