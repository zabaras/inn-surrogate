import argparse
import torch
import json
import random
from pprint import pprint

# always uses cuda if avaliable

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='Solving inverse problem using cINN')
        #Block-1
        self.add_argument('--split_channel', type=list, default=[2,2], help='Input dimension block-1 in C1,C2') 
        self.add_argument('--cond_size', type=int, default=128, help='Conditioning dimension') 
        self.add_argument('--hidden_layer_channel', type=int, default=64, help='number of channels for the hidden layer') 
        #Block-2
        self.add_argument('--input_dimension1_r', type=list, default=[(2,32,32)], help='Input dimension block-2 in CXHXW') 
        self.add_argument('--cond_size1', type=int, default=128, help='Conditioning dimension') 
        self.add_argument('--hidden_layer_channel1', type=int, default=64, help='number of channels for the hidden layer') 
        self.add_argument('--input_dimension1', type=int, default=4, help='coupling block-2') 
        self.add_argument('--input_dimension12', type=int, default=3, help='coupling block-2')      
        self.add_argument('--permute_a1', type=int, default=4, help='permutation for the invertible block-2')        
        #Block-3
        self.add_argument('--input_dimension2_r', type=list, default=[(4,16,16)], help='Input dimension block-3 in CXHXW')
        self.add_argument('--cond_size2', type=int, default=48, help='Conditioning dimension')
        self.add_argument('--hidden_layer_channel2', type=int, default=96, help='number of channels for the hidden layer')
        self.add_argument('--input_dimension2', type=int, default=8, help='invertible block-3')
        self.add_argument('--input_dimension22', type=int, default=3, help='dinvertible block-3')        
        self.add_argument('--permute_a2', type=int, default=8, help='permutation for the invertible block-3')    
        # training
        self.add_argument('--epochs', type=int, default=102, help='number of epochs to train (default: 200)')
        self.add_argument('--lr', type=float, default=0.0008, help='learnign rate')
        self.add_argument('--weight_decay', type=float, default=0.00005, help="weight decay")
        self.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')
        self.add_argument('--test-batch-size', type=int, default=128, help='input batch size for testing (default: 100)')
        self.add_argument('--seed', type=int, default=1, help='manual seed used in Tensor')
        self.add_argument('--ntrain', type=int, default=10000, help="number of training data")
        self.add_argument('--ntest', type=int, default=128, help="number of test data") 
        #Block-4
        self.add_argument('--cond_size3', type=int, default=512, help='Conditioning dimension')
        self.add_argument('--hidden_layer3', type=int, default=4096, help='number of channels for the hidden layer')
        self.add_argument('--input_dimension3', type=int, default=1024, help='invertible block-4')
        self.add_argument('--input_dimension32', type=int, default=1, help='dinvertible block-4')        
        self.add_argument('--permute_a3', type=int, default=1024, help='permutation for the invertible block-4')  


    def parse(self):
        args = self.parse_args()
        # seed
        if args.seed is None:
            args.seed = random.randint(1, 10000)
        print("Random Seed: ", args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        print('Arguments:')
        pprint(vars(args))
        return args

# global
args = Parser().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
