import torch
import torch.nn as nn
from models.conditioning_network import conditioning_network
from models.main_model import main_file
from time import time
import torch.optim as optim
import torch.nn.functional as F
from args import args, device
import sys
import h5py
import os
import numpy as np
import scipy.io as io
from utils.plot import  save_samples
import matplotlib.pyplot as plt
from utils.load_data import load_data
from utils.error_bars import error_bar, train_test_error, plot_std


# load the data here
train_loader, test_loader, sample_loader, NLL_test_loader = load_data()
print('loaded the data.........')



# this is the s and the t network

def convolution_network(Hidden_layer):
    return lambda input_channel, output_channel: nn.Sequential(
                                    nn.Conv3d(input_channel, Hidden_layer, (1,3,3), padding=(0,1,1)),
                                    nn.Dropout3d(p=0.1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(Hidden_layer, output_channel, (1,3,3), padding=(0,1,1)))

def fully_connected(Hidden_layer):
    return lambda input_data, output_data: nn.Sequential(
                                    nn.Linear(input_data, Hidden_layer),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(Hidden_layer, output_data))


network_s_t = convolution_network(args.hidden_layer_channel)    
network_s_t2 = convolution_network(args.hidden_layer_channel2) 
network_s_t3 = fully_connected(args.hidden_layer3)
#load network
INN_network = main_file(args.cond_size,network_s_t,
                    args.input_dimension1,args.input_dimension12,args.cond_size1,args.permute_a1,args.split_channel,args.input_dimension1_r,
                    args.input_dimension2,args.input_dimension22,args.cond_size2,args.permute_a2,network_s_t2,args.input_dimension2_r,
                    args.input_dimension3,args.input_dimension32,args.cond_size3,network_s_t3,args.permute_a3).to(device)
cond_network = conditioning_network().to(device)



combine_parameters = [parameters_net for parameters_net in INN_network.parameters() if parameters_net.requires_grad]
for parameters_net in combine_parameters:
    parameters_net.data = 0.02 * torch.randn_like(parameters_net)



combine_parameters += list(cond_network.parameters())
optimizer = torch.optim.Adam(combine_parameters, lr=args.lr, weight_decay=args.weight_decay)


def train(N_epochs):
    INN_network.train()
    cond_network.train()
    loss_mean = []
    loss_val = []
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.float(), y.float()
        x, y = x.to(device), y.to(device)
        x = x.view(16,1,4,64,64)
        y = y.view(16,4,4,64) # for config_1  change this to y = y.view(16,2,4,64)
        y1 = cond_network(y)
        input = x
        c = y1[2]   
        c2 = y1[1]
        c3 = y1[0]
        c4 = y1[3]
        z,log_j = INN_network(input,c,c2,c3,c4,forward=True)
        loss = torch.mean(z**2) / 2 - torch.mean(log_j) / ( 1 * 4 * 64 * 64)
        loss.backward()      
        loss_mean.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    loss_mean1 = loss_mean
    return loss_mean1

def test(epoch):
    INN_network.eval()
    cond_network.eval()
    loss_mean = []
    loss_val = []
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.float(), target.float()
        x, y = input.to(device), target.to(device)
        x = x.view(16,1,4,64,64)
        y = y.view(16,4,4,64)# for config_1  change this to y = y.view(16,2,4,64)
        input, target = x.view(16,1,4,64,64), y.view(16,4,4,64)  # for config_1  change this to target = y.view(16,2,4,64)
        x = input.view(16,1,4,64,64)
        y = target.view(16,4,4,64)# for config_1  change this to y = target.view(16,2,4,64)
        tic = time()
        y1 = cond_network(y)
        c = y1[2]   
        c2 = y1[1]
        c3 = y1[0]
        c4 = y1[3]
        z,log_j = INN_network(x,c,c2,c3,c4,forward=True)
        loss_val = torch.mean(z**2) / 2 - torch.mean(log_j) /( 1 * 4 * 64 * 64)
        loss_mean.append(loss_val.item())
    loss_mean1 = loss_mean
    return loss_mean1



def sample2(epoch):
    INN_network.eval()
    cond_network.eval()
    loss_mean = []
    loss_val = []
    for batch_idx, (input, target) in enumerate(sample_loader):
        input, target = input.float(), target.float()
        x, y = input.to(device), target.to(device)
        x = x.view(1,1,4,64,64)
        y = y.view(1,4,4,64) # for config_1  change this to y = y.view(1,2,4,64)
        input, target = x.view(1,1,4,64,64), y.view(1,4,4,64) # for config_1  change this to target = y.view(1,2,4,64)
        x = input.view(1,1,4,64,64)
        y = target.view(1,4,4,64) # for config_1  change this to y = target.view(16,2,4,64)
        labels_test = target   
        N_samples = 1000
        labels_test = labels_test[0,:,:,:]
        labels_test = labels_test.cpu().data.numpy()
        l = np.repeat(np.array(labels_test)[np.newaxis,:,:,:], N_samples, axis=0)
        l = torch.Tensor(l).to(device)            
        z = torch.randn(N_samples,16384).to(device)
        with torch.no_grad():
            y1 = cond_network(l)
            input = x.view(1,1,4,64,64)
            c = y1[2]   
            c2 = y1[1]
            c3 = y1[0]
            c4 = y1[3]
            val = INN_network(z,c,c2,c3,c4,forward=False)
        rev_x = val.cpu().data.numpy()
        if epoch == 200:
            input_test = input.cpu().data.numpy()
            f2 = h5py.File('data_save_%d.h5'%epoch, 'w')
            f2.create_dataset('input', data=input_test, compression='gzip', compression_opts=9)
            f2.create_dataset('pred', data=rev_x, compression='gzip', compression_opts=9)
            f2.close()
        if epoch % 20 == 0:
            input_test = input.cpu().data.numpy()
            input1 = input_test.reshape(1,1,4,64,64)
            samples1 = rev_x
            mean_samples1 = np.mean(samples1,axis=0)
            mean_samples1 = mean_samples1.reshape(1,1,4,64,64) # mean of all the samples
            samples1 = samples1[:2,:,:,:,:]
            #=====
            mean_samples_layer1 = mean_samples1[0,0,1,:,:]
            mean_samples_layer1 = mean_samples_layer1.reshape(1,1,64,64)
            input_layer1 = input1[0,0,1,:,:]
            input_layer1 = input_layer1.reshape(1,1,64,64)
            samples_layer1 = samples1[:,0,1,:,:]
            samples_layer1 = samples_layer1.reshape(2,1,64,64)
            x1 = np.concatenate((input_layer1,mean_samples_layer1,samples_layer1),axis=0)
            actual = input_layer1
            pred = rev_x[:,:,1,:,:]
            pred = pred.reshape(1000,1,64,64)
            error_bar(actual,pred,epoch,1)

#==========================================================
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
#==========================================================


print('training start .............')
mkdir('results')
loss_train_all = []
loss_test_all = []
tic = time()
for epoch in range(1,args.epochs):
    print('epoch number .......',epoch)
    loss_train = train(epoch)
    loss_train2 = np.mean(loss_train)
    loss_train_all.append(loss_train2)
    with torch.no_grad():
        sample2(epoch)
        loss_test = test(epoch)
        loss_test = np.mean(loss_test)
        print(('NLL loss:',loss_test))
        loss_test_all.append(loss_test)

epoch1 = 200
torch.save(INN_network.state_dict(), f'INN_network_epoch{epoch1}.pt')
torch.save(cond_network.state_dict(), f'cond_network_epoch{epoch1}.pt')
loss_train_all = np.array(loss_train_all)
loss_test_all = np.array(loss_test_all)
print('saving the training error and testing error')
io.savemat('training_loss.mat', dict([('training_loss',np.array(loss_train_all))]))
io.savemat('test_loss.mat', dict([('testing_loss',np.array(loss_test_all))]))
print('plotting the training error and testing error')
train_test_error(loss_train_all,loss_test_all, epoch1)
toc = time()
print('total traning taken:', toc-tic)


domain = 16384
def test_NLL():
    INN_network.eval()
    cond_network.eval()
    loss_mean = []
    loss_val = []
    final_concat = []
    for batch_idx, (input, target) in enumerate(NLL_test_loader):
        input, target = input.float(), target.float()
        x, y = input.to(device), target.to(device) 
        input, target = x.view(128,1,4,64,64), y.view(128,4,4,64) # for config_1  change this to target = y.view(16,2,4,64)
        labels_test1 = target
        N_samples = 1000

        for jj in range(128):
            labels_test = labels_test1[jj,:,:,:]
            x = input[jj,:,:,:,:]
            labels_test = labels_test.cpu().data.numpy()
            l = np.repeat(np.array(labels_test)[np.newaxis,:,:,:], N_samples, axis=0)
            l = torch.Tensor(l).to(device)            
            z = torch.randn(N_samples,16384).to(device)
            with torch.no_grad():
                y1 = cond_network(l)
                c = y1[2]   
                c2 = y1[1]
                c3 = y1[0]
                c4 = y1[3]
                val = INN_network(z,c,c2,c3,c4,forward=False)
            rev_x = val.cpu().data.numpy()
            input1 = x.cpu().data.numpy()
            input1 = input1.reshape(1,4,64,64)
            rev_x = rev_x.reshape(1000,1,4,64,64)
            mean_val = rev_x.mean(axis=0)
            mean_val = mean_val.reshape(1,4,64,64)
            d1 = (1/domain)*np.sum(input1**2)
            n1 = (1/domain)*np.sum((input1-mean_val)**2)
            m1 = n1/d1
            final_concat.append(m1)
        final_concat = np.array(final_concat)
    return final_concat


with torch.no_grad():
    final_error = test_NLL()
    old_val = np.mean(final_error)
    print('NRMSE:',np.mean(final_error))
