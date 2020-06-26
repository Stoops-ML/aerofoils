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

        # paths and files
        self.root_dir = Path(root_dir)
        self.aerofoils = [file for file in os.listdir(root_dir) if re.search(r"(.csv)$", file)]

        # initialise variables
        self.x = [[] for _ in range(len(self.aerofoils))]  # input: coordinates of aerofoil
        self.y = [[] for _ in range(len(self.aerofoils))]  # outputs: max ClCd at angle (2 outputs)
        self.aerofoil = [None for _ in range(len(self.aerofoils))]

        # transforms
        self.transform = transform

        # tensor sizes
        self.num_channels = num_channels
        self.output_size = output_size
        self.input_size = input_size

    def __getitem__(self, item):
        """index into dataset"""
        if torch.is_tensor(item):
            item = item.tolist()

        # get max ClCd at angle
        with open(self.root_dir / self.aerofoils[item]) as f:
            line = f.readline()
            y_vals = [float(num) for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
            max_ClCd, angle = y_vals[0], y_vals[1]

        # get coordinates
        coords = np.loadtxt(self.root_dir / self.aerofoils[item], delimiter=" ", dtype=np.float32, skiprows=1)

        # organise data
        self.x[item] = np.array(coords[:, 1], dtype=np.float32)  # inputs as ndarrays
        self.y[item] = np.array([max_ClCd, angle], dtype=np.float32)  # outputs as ndarrays
        self.aerofoil[item] = self.aerofoils[item]

        # convert data to tensor with correct shape
        self.x[item] = torch.from_numpy(self.x[item]).view(self.num_channels, self.input_size)
        self.y[item] = torch.from_numpy(self.y[item]).view(1, self.output_size)

        if self.transform:
            self.y[item] = self.transform(self.y[item])  # only transform input (for now)

        # return must be in this form for dataloader to be an iterator (therefore can't be a dictionary)
        return self.x[item], self.y[item], self.aerofoil[item]

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)


class FlipHorizontalRandom:
    def __call__(self, sample, thresh=0.8):
        """randomly flips the aerofoil horizontally"""
        if random() > thresh:
            sample = torch.flip(sample, [0, 1])
        return sample


class NormaliseYValues:
    def __init__(self, dir):
        """find mean and standard deviation of all 'aerofoils' in directory 'dir' """

        aerofoils = [file for file in os.listdir(dir) if re.search(r"(.csv)$", file)]

        # get all output values: max ClCd, angle
        ClCd_list = []
        angle_list = []
        for aerofoil in aerofoils:
            with open(dir / aerofoil) as f:
                outputs = [num for num in re.findall(r'[+-]?\d*[.]?\d*', f.readline()) if num != '']
                ClCd_list.append(float(outputs[0]))
                angle_list.append(float(outputs[1]))

        # calculate mean
        self.ClCd_mean = sum(ClCd_list) / len(aerofoils)
        self.angle_mean = sum(angle_list) / len(aerofoils)

        # calculate standard deviation
        ClCd_list_SD = [(ClCd - self.ClCd_mean) ** 2 for ClCd in ClCd_list]
        angle_list_SD = [(angle - self.angle_mean) ** 2 for angle in angle_list]
        self.ClCd_SD = np.sqrt(sum(ClCd_list_SD) / len(aerofoils))
        self.angle_SD = np.sqrt(sum(angle_list_SD) / len(aerofoils))

    def __call__(self, outputs):
        """normalise y values of sample 'sample'.
         mean 0 and variance 1 (best for backprop to have range of [-1,1], not [0,1])"""
        outputs[0, 0] = (outputs[0, 0] - self.ClCd_mean) / self.ClCd_SD  # normalise max ClCd
        outputs[0, 1] = (outputs[0, 1] - self.angle_mean) / self.angle_SD  # normalise angle
        return outputs
