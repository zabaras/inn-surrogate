import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

ntrain = 10000

def load_data():
    Train_hdf5_file ='Config_2_train_obs_1pc_3D.hdf5'
    with h5py.File(Train_hdf5_file, 'r') as f:
        x_train = f['input'][:ntrain]
        y_train = f['output'][:ntrain]
        print('x_train:',x_train.shape)
        print('y_train:',y_train.shape)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(x_train),torch.FloatTensor(y_train)), batch_size=16, shuffle=True, drop_last=True)



    Test_hdf5_file ='Config_2_test_obs_1pc_3D.hdf5'
    with h5py.File(Test_hdf5_file, 'r') as f1:
        x_test = f1['input']
        y_test_new = f1['output']
        print('x_test:',x_test.shape)
        print('y_test:',y_test_new.shape)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(x_test),torch.FloatTensor(y_test_new)),batch_size=16, shuffle=False, drop_last=True)
        test_loader_nll = DataLoader(TensorDataset(torch.FloatTensor(x_test),torch.FloatTensor(y_test_new)),batch_size=128, shuffle=False, drop_last=True)


    Sample_hdf5_file ='Config_2_sample_obs_1pc_3D.hdf5'
    with h5py.File(Sample_hdf5_file, 'r') as f2:
        x_test = f2['input']
        y_test_new = f2['output']
        print('x_sample:',x_test.shape)
        print('y_sample:',y_test_new.shape)
        sample_loader = DataLoader(TensorDataset(torch.FloatTensor(x_test),torch.FloatTensor(y_test_new)),batch_size=1, shuffle=False, drop_last=True)
    # To load config-1 the make the channels as 1 for the observations as (B,2,obs) 
    # For the train data: y_train_new_config_1 = y_train[:,:2,:,:]
    # For the test data: y_test_new_config_1 = y_test_new[:,:2,:,:] 
    return train_loader,test_loader, sample_loader, test_loader_nll

