import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import re
import numpy as np


class AerofoilDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """data loading"""
        self.root_dir = Path(root_dir)
        self.aerofoils = [file for file in os.listdir(root_dir)
                          if re.search(r"(.csv)$", file)
                          if os.path.isfile(root_dir / file)]
        self.transform = transform

    def __getitem__(self, item):
        """index into dataset"""
        if torch.is_tensor(item):
            item = item.tolist()

        # read data (saves memory by not reading data in __init__)
        # x = []
        y = []
        with open(self.root_dir / self.aerofoils[item]) as f:
            for line in f:
                if re.search(r'ClCd', line):  # find y values
                    y_vals = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
                    max_ClCd, angle = float(y_vals[0]), float(y_vals[1])
                    continue
                xy = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
                # x.append(float(xy[0]))
                y.append(float(xy[1]))
        # coords = np.stack((x, y), axis=0)

        # only taking y coordinates as all aerofoils have same x coordinates
        sample = {"aerofoil": self.aerofoils[item], "coordinates": np.array(y), "y": [max_ClCd, angle]}
        # sample = {"aerofoil": self.aerofoils[item], "coordinates": coords, "y": [max_ClCd, angle]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)

    def get_sizes(self):
        num_channels = len(self.__getitem__(0)['coordinates'].shape)  # number of columns of input (e.g x,y coords)
        if num_channels == 1:  # 1 channel
            input_size = len(self.__getitem__(0)['coordinates'])
        else:  # several channels
            _, input_size = self.__getitem__(0)['coordinates'].shape
        output_size = len(self.__getitem__(0)['y'])

        return num_channels, input_size, output_size


class ToTensor(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample): return {'aerofoil': sample['aerofoil'],
                                        'coordinates': torch.from_numpy(sample['coordinates']),
                                        # 'x': torch.FloatTensor(sample['x']),
                                        'y': torch.FloatTensor(sample['y'])}
