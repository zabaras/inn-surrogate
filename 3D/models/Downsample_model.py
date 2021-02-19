from math import exp
import numpy as np
import torch
import torch.nn as nn
import sys

class Downsample(nn.Module):
    '''
    Args: 
    Input: BXCXDXHXW
    Reference: Jacobsen et al.,"i-revnet: Deep invertible networks." for downsampling.
    '''
    def __init__(self):
        super(Downsample,self).__init__()
    def forward(self, x, sample_the_data=False):
        if sample_the_data == True:
            (batch_size, channel_1,depth_1, height_1, width_1) = x.size()
            channel_2 = channel_1 / 4
            width_2 = width_1 * 2
            height_2 = (height_1) * 2
            depth_2 = depth_1  
            data = x.permute(0, 2, 3, 4, 1)
            data_mod = data.reshape(batch_size, depth_1, height_1, width_1, 4, int(channel_2))
            stack = [data_s.reshape(batch_size, depth_1, height_1, int(width_2),int(channel_2)) for data_s in data_mod.split(2, 4)]
            data = torch.stack(stack, 0).transpose(0, 1)
            data = data.permute(0, 2, 3, 1, 4, 5)
            output = data.reshape(batch_size, int(depth_2), int(height_2), int(width_2), int(channel_2)).permute(0, 4, 1, 2, 3)
            return output
        else:
            (batch_size, channel_2, depth_2, height_2, width_2) = x.size()
            height_1 = height_2 / 2
            channel_1 = channel_2 * 4
            depth_1 = depth_2 
            data = x.permute(0,2,3,4,1)
            stack = [data_s.reshape(batch_size, depth_1, int(height_1), channel_1)for data_s in data.split(2, 3)] #split in the width axis (length=2 and axis=3 which is width)
            data = torch.stack(stack, 2).permute(0, 1, 3, 2, 4)  #NXDXWXHXC
            output = data.permute(0, 4, 1, 2, 3) 
            return output

