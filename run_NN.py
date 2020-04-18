import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import re
import sys


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
        coords = np.loadtxt(self.root_dir / self.aerofoils[item], dtype=np.float32)  # no header in file
        sample = {"aerofoil": self.aerofoils[item], "coordinates": coords}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)


def show_aerofoil(aerofoil, coordinates):
    """show plot of aerofoil"""
    plt.plot(coordinates[:, 0], coordinates[:, 1], 'r-')
    plt.title(aerofoil)
    plt.show()


def show_aerofoil_batch(sample_batched, batch_num):
    """show plot of aerofoils for a batch of samples."""
    aerofoils_batch, coordinates_batch = sample_batched['aerofoil'], sample_batched['coordinates']
    batch_size = len(aerofoils_batch)

    fig = plt.figure()
    for i, (aerofoil, coords) in enumerate(zip(aerofoils_batch, coordinates_batch)):
        ax = fig.add_subplot(1, batch_size, i+1)
        ax.plot(coords[:, 0], coords[:, 1], 'r-')
        ax.title.set_text(aerofoil)

    plt.suptitle(f'Batch #{batch_num} from dataloader')
    plt.show()


class ToTensor(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample): return {'aerofoil': sample['aerofoil'],
                                        'coordinates': torch.from_numpy(sample['coordinates'])}


# file configuration
path = Path(__file__).parent

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 69  # number of coordinates of x,y in all aerofoils
hidden_size = 100
num_epochs = 2
batch_size = 4
learning_rate = 0.001

# import dataset
train_dataset = AerofoilDataset(path / 'data' / 'out' / 'train', transform=transforms.Compose([ToTensor()]))
# note: don't actually need to do the ToTensor transform because this is already done by the dataloader. It's needed for
# images where you need to switch axes
valid_dataset = AerofoilDataset(path / 'data' / 'out' / 'valid')
# show_aerofoil(**train_dataset[0])

# dataloader
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
for i, batch in enumerate(dataloader):
    show_aerofoil_batch(batch, i)
