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
        coords = np.loadtxt(self.root_dir / self.aerofoils[item], dtype=np.float32, skiprows=1)
        with open(self.root_dir / self.aerofoils[item]) as f:
            found_ClCd = False
            for line in f:
                if re.search(r'ClCd', line):
                    obj = re.search(r'\D*([+-]?\d*[.]?\d*)\D+([+-]?\d*[.]?\d*)', line)
                    found_ClCd = True
                    break
            if not found_ClCd:
                raise Exception(f"no Max ClCd & angle in file {self.aerofoils[item]}. Code ended")
            max_ClCd, angle = float(obj.group(1)), float(obj.group(2))

        # TODO: coordinates of sample is redundant (we have x for this). Remove and update other classes
        sample = {"aerofoil": self.aerofoils[item], "coordinates": coords, "y": [max_ClCd, angle]}  #,
                  # "x": np.append(coords[:, 0], coords[:, 1]), "y": [max_ClCd, angle]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """get length of dataset"""
        return len(self.aerofoils)


def show_aerofoil(**kwargs):
    """show plot of aerofoil"""
    plt.plot(kwargs["coordinates"][:, 0], kwargs["coordinates"][:, 1], 'r-')
    plt.title(kwargs["aerofoil"])
    ClCd, angle = kwargs["y"]
    plt.text(0, 0, f"Max ClCd = {ClCd:.2f} at {angle:.2f} degrees")
    plt.show()


def show_aerofoil_batch(batch_num, **sample_batched):
    """show plot of aerofoils for a batch of samples."""
    aerofoils_batch, coordinates_batch, y_batch = sample_batched['aerofoil'], sample_batched['coordinates'],\
                                                  sample_batched["y"]
    ClCd_batch, angle_batch = y_batch[:, 0], y_batch[:, 1]
    batch_size = len(aerofoils_batch)

    fig = plt.figure()
    for i, (aerofoil, coords, ClCd, angle) in enumerate(zip(aerofoils_batch, coordinates_batch, ClCd_batch,
                                                            angle_batch)):
        ax = fig.add_subplot(1, batch_size, i+1)
        ax.plot(coords[:, 0], coords[:, 1], 'r-')
        ax.text(0, 0, f"Max ClCd = {ClCd:.2f}\nat {angle:.2f}deg")
        ax.title.set_text(aerofoil)

    plt.suptitle(f'Batch #{batch_num} from dataloader')
    plt.show()


class ToTensor(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample): return {'aerofoil': sample['aerofoil'],
                                        'coordinates': torch.from_numpy(sample['coordinates']),
                                        # 'x': torch.FloatTensor(sample['x']),
                                        'y': torch.FloatTensor(sample['y'])}


# file configuration
path = Path(__file__).parent

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 69*2  # number of coordinates of x,y in all aerofoils
output_size = 2  # max ClCd & angle
hidden_size = 100
num_epochs = 1000
bs = 4
learning_rate = 0.01

# TODO: find input and output size from files automatically


# import dataset
train_dataset = AerofoilDataset(path / 'data' / 'out' / 'train', transform=transforms.Compose([ToTensor()]))
test_dataset = AerofoilDataset(path / 'data' / 'out' / 'valid', transform=transforms.Compose([ToTensor()]))
# note: don't actually need to do the ToTensor transform because this is already done by the dataloader. It's needed for
# images where you need to switch axes
# show_aerofoil(**train_dataset[0])

# dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=4)
# for i, batch in enumerate(test_loader):
#     show_aerofoil_batch(i, **batch)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_outputs):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_layers[0])
        self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.linear3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.linear4 = nn.Linear(hidden_layers[2], num_outputs)
        self.relu = nn.ReLU()

        # TODO: use nn.ModuleDict to iterate through layers
        # self.nn_model = {}
        # for i, layer_size in enumerate(layers):
        #     self.nn_model[f"layer{i+1}"] = nn.Linear(layer_size, layers[i+1])
        #     if i+1 == len(layers)-1:
        #         break

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)

        # out = x
        # for i in range(len(self.nn_model)):
        #     out = self.nn_model[f"layer{i+1}"](out)
        #     if i+1 != len(self.nn_model):  # don't do activation on last layer
        #         print("activation applied")
        #         out = self.relu(out)

        # I think MSEloss() applies an activation function to the last layer itself
        return out


hidden_layers = [hidden_size] * 4
model = NeuralNet(input_size, hidden_layers, output_size)

# loss and optimiser
criterion = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for i, sample in enumerate(train_loader):
        # reshape input to column vector
        sample["coordinates"] = sample["coordinates"].reshape(-1, input_size).to(device)
        sample["y"] = sample["y"].to(device)

        # forward pass
        pred_output = model(sample["coordinates"])
        loss = criterion(pred_output, sample["y"])

        # backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()  # do an update step to update parameters

        # print output
        if (epoch+1) % 10 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}. Loss = {loss.item():.4f}")

# test
with torch.no_grad(): # don't add gradients of test set to computational graph
    for sample_batched in test_loader:
        sample_batched["coordinates"] = sample_batched["coordinates"].reshape(-1, input_size).to(device)
        sample_batched["y"] = sample_batched["y"].to(device)
        pred_output = model(sample_batched["coordinates"])
        for aerofoil, prediction in zip(sample_batched['aerofoil'], pred_output):
            print(f"Aerofoil {aerofoil}: Max ClCd = {prediction[0]:.2f} at {prediction[1]:.2f}deg")

