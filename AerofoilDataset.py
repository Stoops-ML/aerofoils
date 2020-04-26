import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import re
import numpy as np
import sys


class AerofoilDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """data loading"""
        self.root_dir = Path(root_dir)
        self.aerofoils = [file for file in os.listdir(root_dir)
                          if re.search(r"(.csv)$", file)]
        self.transform = transform
        self.x = [[] for _ in range(len(self.aerofoils))]  # input: coordinates of aerofoil
        self.y = [[] for _ in range(len(self.aerofoils))]  # outputs: max ClCd at angle (2 outputs)

    def __getitem__(self, item):
        """index into dataset"""
        if torch.is_tensor(item):
            item = item.tolist()

        with open(self.root_dir / self.aerofoils[item]) as f:
            line = f.readline()
            y_vals = [float(num) for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
            max_ClCd, angle = y_vals[0], y_vals[1]

        coords = np.loadtxt(self.root_dir / self.aerofoils[item], delimiter=" ", dtype=np.float32, skiprows=1)
        self.x[item] = np.array(coords[:, 1], dtype=np.float32)  # input
        self.y[item] = np.array([max_ClCd, angle], dtype=np.float32)  # output

        if self.transform:
            self.x[item] = self.transform(self.x[item])
            self.y[item] = self.transform(self.y[item])

        return self.x[item], self.y[item]  # return must be in this form for dataloader to be an iterator (therefore can't be a dictionary)

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)

    def get_sizes(self):
        num_channels = len(self.__getitem__(0)[0].shape)
        if num_channels == 1:  # 1 channel
            input_size = len(self.__getitem__(0)[0])
        else:  # several channels
            _, input_size = self.__getitem__(0)[0].shape
        output_size = len(self.__getitem__(0)[1])

        return num_channels, input_size, output_size


class ToTensor(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)
