import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import sys


class AerofoilDataset(Dataset):
    def __init__(self, root_dir):
        """data loading"""
        self.root_dir = Path(root_dir)
        self.aerofoils = [file for file in os.listdir(root_dir) if 'csv' in file
                          if os.path.isfile(root_dir / file)]

    def __getitem__(self, item):
        """index into dataset"""
        if torch.is_tensor(item):
            item = item.tolist()

        aerofoil_name = os.listdir(self.root_dir)[item]
        coords = np.loadtxt(self.root_dir / aerofoil_name, dtype=np.float32)  # no header in file, delimiter is space

        sample = {"aerofoil": aerofoil_name, "coordiantes": coords}

        return sample

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)


def show_aerofoil(aerofoil, coordiantes):
    """show plot of aerofoil"""
    plt.plot(coordiantes[:, 0], coordiantes[:, 1], 'r-')
    plt.title(aerofoil)
    plt.show()


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 69  # number of coordinates of x,y in all aerofoils
hidden_size = 100
num_epochs = 2
batch_size = 4
learning_rate = 0.001
path = Path(__file__).parent

# import dataset
train_dataset = AerofoilDataset(path / 'data' / 'out')
show_aerofoil(**train_dataset[0])


