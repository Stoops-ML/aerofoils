import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 69  # number of coordinates of x,y in all aerofoils
hidden_size = 100
num_epochs = 2
batch_size = 4
learning_rate = 0.001

# import dataset
train_dataset =


class AerofoilDataset(Dataset):
    def __init__(self, root_dir, max_clcld_fname):
        """data loading"""
        self.root_dir = root_dir
        self.aerofoil_fnames = [file for file in os.listdir(root_dir) if not max_clcld_fname in file]


    def __getitem__(self, item):
        """index into dataset"""
        if torch.is_tensor(item):
            item - item.tolist()

        aerofoil_name = os.path.join(self.root_dir, self.aerofoil_fnames[item + '.csv'])
        coords = np.loadtxt(aerofoil_name, delimeter=' ', dtype=np.float32, skiprows=1)


    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoil_fnames)