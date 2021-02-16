from math import exp
import numpy as np
import torch
import torch.nn as nn

class CouplingBlock(nn.Module):
    '''
    Args:
    s_t_network: scale and shit network
    input_dimension_1: Input dimension
    input_dimension_2: length of the input 
    We use soft clamp as menioned in https://arxiv.org/abs/1907.02392 (Reference)

    '''
    def __init__(self, s_t_network, input_dimension_1,input_dimension_2, condition_dimension):
        super().__init__()
        self.channel_part_1 = input_dimension_1 // 2
        self.channel_part_2 = input_dimension_1 - input_dimension_1 // 2
        self.s_net = s_t_network(self.channel_part_1 + condition_dimension, self.channel_part_1)
        self.t_net = s_t_network(self.channel_part_2 + condition_dimension, self.channel_part_2)
        self.input_len = input_dimension_2


    def jacobian(self):
        jacobian_val = self.jacobian_output
        return jacobian_val

    def forward(self, x, c, sample_the_data=False):
        x1 = x.narrow(1, 0, self.channel_part_1)
        x2 = x.narrow(1, self.channel_part_1, self.channel_part_2)

        if sample_the_data == False:
            x1_c = torch.cat([x1, c], 1) 
            self.s_network = self.s_net(x1_c)
            self.t_network = self.t_net(x1_c)
            y2 = (torch.exp(0.636 *2* torch.atan(self.s_network))) * x2 + self.t_network
            output = torch.cat((x1, y2), 1)
            jacobian2 = torch.sum((0.636 *2* torch.atan(self.s_network)), tuple(range(1, self.input_len+1)))
            self.jacobian_output = jacobian2
            return output
        else:
            x1_c = torch.cat([x1, c], 1) 
            self.s_network = self.s_net(x1_c)
            self.t_network = self.t_net(x1_c)
            temp = (x2 - self.t_network) / (torch.exp(0.636 *2* torch.atan(self.s_network)))
            output = torch.cat((x1, temp), 1)
            jacobian1 = torch.sum((0.636 *2* torch.atan(self.s_network )), dim=tuple(range(1, self.input_len+1)))
            self.jacobian_output = (- jacobian1)
            return output


