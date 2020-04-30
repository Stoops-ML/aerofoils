import torch
from torch.utils.data import Dataset
from pathlib import Path
import os
import re
import numpy as np
from random import random
import sys


class AerofoilDataset(Dataset):
    def __init__(self, root_dir, num_channels, input_size, output_size, transform=None):
        """data loading"""
        self.root_dir = Path(root_dir)
        self.aerofoils = [file for file in os.listdir(root_dir)
                          if re.search(r"(.csv)$", file)]
        self.transform = transform
        self.x = [[] for _ in range(len(self.aerofoils))]  # input: coordinates of aerofoil
        self.y = [[] for _ in range(len(self.aerofoils))]  # outputs: max ClCd at angle (2 outputs)
        self.aerofoil = [None for _ in range(len(self.aerofoils))]

        self.num_channels = num_channels
        self.output_size = output_size
        self.input_size = input_size

    def __getitem__(self, item):
        """index into dataset"""
        if torch.is_tensor(item):
            item = item.tolist()

        # get coordinates and max ClCd at angle
        with open(self.root_dir / self.aerofoils[item]) as f:
            line = f.readline()
            y_vals = [float(num) for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
            max_ClCd, angle = y_vals[0], y_vals[1]
        coords = np.loadtxt(self.root_dir / self.aerofoils[item], delimiter=" ", dtype=np.float32, skiprows=1)

        # organise data
        self.x[item] = np.array(coords[:, 1], dtype=np.float32)  # inputs as ndarrays
        self.y[item] = np.array([max_ClCd, angle], dtype=np.float32)  # outputs as ndarrays
        self.aerofoil[item] = self.aerofoils[item]

        # convert data to tensor with correct shape
        self.x[item] = torch.from_numpy(self.x[item]).view(self.num_channels, self.input_size)
        self.y[item] = torch.from_numpy(self.y[item]).view(1, self.output_size)

        if self.transform:
            self.x[item] = self.transform(self.x[item])  # only transform input

        # return must be in this form for dataloader to be an iterator (therefore can't be a dictionary)
        return self.x[item], self.y[item], self.aerofoil[item]

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)


class ToTensor:
    def __call__(self, sample):
        """convert ndarrays in sample to Tensors."""
        return torch.from_numpy(sample)


class FlipHorizontalRandom:
    def __call__(self, sample, thresh=0.8):
        """randomly flips the aerofoil horizontally"""
        if random() > thresh:
            sample = torch.flip(sample, [0, 1])
        return sample
