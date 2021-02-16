import numpy as np
import torch
import torch.nn as nn
class divide_data(nn.Module):
    '''Args:
        X: input (BXD) to  output (BXCXHXW) 
        (This is used to split the data  for the concat part for Z and 
        the other part for the network during the training and sampling phase).
    '''
    def __init__(self, input_dimension, split_data_channel):
        super(divide_data,self).__init__()
        self.split_data_channel = split_data_channel
    def forward(self, x, sample_the_data=False):
        out = torch.split(x, self.split_data_channel,1)
        return out
