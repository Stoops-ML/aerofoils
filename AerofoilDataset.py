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
        self.aerofoil = [None for _ in range(len(self.aerofoils))]

        self.num_channels = 1  # one channel for y coordinate (xy coordinates requires two channels)
        self.output_size = 0
        self.input_size = 0

    def __getitem__(self, item):
        """index into dataset"""
        if torch.is_tensor(item):
            item = item.tolist()

        with open(self.root_dir / self.aerofoils[item]) as f:
            line = f.readline()
            y_vals = [float(num) for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
            max_ClCd, angle = y_vals[0], y_vals[1]

        coords = np.loadtxt(self.root_dir / self.aerofoils[item], delimiter=" ", dtype=np.float32, skiprows=1)
        self.x[item] = torch.from_numpy(np.array(coords[:, 1], dtype=np.float32)
                                        ).view(self.num_channels, self.input_size)  # inputs
        self.y[item] = torch.from_numpy(np.array([max_ClCd, angle], dtype=np.float32)
                                        ).view(1, self.output_size)  # outputs
        self.aerofoil[item] = self.aerofoils[item]

        if self.transform:
            self.x[item] = self.transform(self.x[item])
            self.y[item] = self.transform(self.y[item])

        # return must be in this form for dataloader to be an iterator (therefore can't be a dictionary)
        return self.x[item], self.y[item], self.aerofoil[item]

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)

    def get_sizes(self):
        coords = np.loadtxt(self.root_dir / self.aerofoils[0], delimiter=" ", dtype=np.float32, skiprows=1)
        self.input_size = len(coords)

        with open(self.root_dir / self.aerofoils[0]) as f:
            line = f.readline()
            y_vals = [float(num) for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
            self.output_size = len(y_vals)

        return self.num_channels, self.input_size, self.output_size


class ToTensor(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)
