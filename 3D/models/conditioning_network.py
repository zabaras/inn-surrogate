import torch
import torch.nn as nn
import torch.optim
import numpy as np 

class conditioning_network(nn.Module):
    '''conditioning network
        The input to the conditioning network are the observations (y)
        Args: 
        y: Observations (B X Obs)
    '''
    def __init__(self):
        super().__init__()

        class flat_data(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):
                return x.view(x.shape[0], -1)

        class unflat_data(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def forward(self, x):

                if x[:,0,0,0].shape == (1,):
                    out = x.view(1,4,4,8,8) # for config_1  change this to out = x.view(1,2,4,8,8)
                if x[:,0,0,0].shape == (16,):
                    out = x.view(16,4,4,8,8) # for config_1  change this to out = x.view(16,2,4,8,8)
                elif x[:,0,0,0].shape == (1000,):
                    out = x.view(1000,4,4,8,8) # for config_1  change this to out = x.view(1000,2,4,8,8)
                return out

        self.multiscale = nn.ModuleList([
                           nn.Sequential(unflat_data(),
                                         nn.ConvTranspose3d(4,  48, (1,2,2), padding=0), # for config_1  change this to nn.ConvTranspose3d(2,  48, (1,2,2), padding=0)
                                         nn.ReLU(inplace=True),
                                         nn.ConvTranspose3d(48, 48, (1,2,2), padding=(0,1,1),stride=(1,2,2))),
                                         nn.Sequential(nn.ReLU(inplace=True),
                                         nn.ConvTranspose3d(48,  64, (1,2,2), padding=0,stride=(1, 1, 1)),
                                         nn.ReLU(inplace=True),
                                         nn.ConvTranspose3d(64, 64, (1,2,2), padding=(0,1,1), stride=(1,2,2))),
                                         nn.Sequential(nn.ReLU(inplace=True),
                                         nn.ConvTranspose3d(64, 64, (1,2,2), padding=0, stride=(1,2,2)),
                                         nn.ReLU(inplace=True),
                                         nn.ConvTranspose3d(64, 64, (1,1,1), padding=0, stride=1)),
                                         nn.Sequential(nn.ReLU(inplace=True),
                                         nn.AvgPool3d(4),
                                         flat_data(),
                                         nn.Linear(16384, 12288),
                                         nn.ReLU(inplace=True), 
                                         nn.Linear(12288, 6144),
                                         nn.ReLU(inplace=True), 
                                         nn.Linear(6144, 5000),
                                         nn.ReLU(inplace=True),                                        
                                         nn.Linear(5000, 4096))])
                                         

    def forward(self, cond):
        value = [cond]
        for value_data in self.multiscale:
            value.append(value_data(value[-1]))
        return value[1:]

