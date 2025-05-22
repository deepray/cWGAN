# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: April 10, 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import matplotlib.image as mpimg

class LoadDataset(Dataset):
    '''
    Loading dataset from list of filenames. The first file needs to
    correspond to X and the seconf to Y
    '''
    def __init__(self, datafiles,N=1000):
        self.data_x  = self.load_data(datafiles[0],N)
        self.data_y  = self.load_data(datafiles[1],N)
        self.x_shape  = self.data_x.shape
        self.y_shape  = self.data_y.shape

        print(f'     *** Datasets:')
        print(f'         ... samples loaded   = {N}')
        print(f'         ... X dimension      = {self.x_shape[1]}X{self.x_shape[2]}X{self.x_shape[3]}')
        print(f'         ... Y dimension      = {self.y_shape[1]}X{self.y_shape[2]}X{self.y_shape[3]}')

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data_x[idx]
        y = self.data_y[idx]
    
        return x,y

    
    # Assuming loaded file is ".npy". If not, then change the function below
    # We need to make sure that the second dimension of the array is the 
    # number of channels. If the last dimension is channels, feed 
    # permute=True to switch the dimensions
    def load_data(self,datafile,N,permute=False):

        try:
            all_data = np.load(datafile).astype(np.float32)
        except FileNotFoundError:
            print(f"Error: The file '{datafile}' was not found.")

        # Hacky permute
        if permute:
            all_data = all_data.permute(0,3,1,2) 

        N_,C_,W_,H_ = all_data.shape

        assert N <= N_
        
        data = torch.tensor(all_data[0:N])

        return data







