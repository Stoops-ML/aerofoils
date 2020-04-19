import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
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
                    obj = re.findall(r'[+-]?\d*[.]?\d*', line)
                    obj = [num for num in obj if num != '']
                    found_ClCd = True
                    break
            if not found_ClCd:
                raise Exception(f"no Max ClCd & angle in file {self.aerofoils[item]}. Code ended")
            max_ClCd, angle = float(obj[0]), float(obj[1])

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
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
hidden_layers = [300, 300, 300, 300, 300, 300, 300, 300]
num_epochs = 10000
bs = 10
learning_rate = 0.001
train_dir = path / 'data' / 'out' / 'train'
test_dir = path / 'data' / 'out' / 'test'

# find input & output size
input_file = os.listdir(train_dir)[0] if re.search(r"(.csv)$", os.listdir(train_dir)[0]) else os.listdir(train_dir)[1]
with open(train_dir / input_file) as f:
    obj = re.findall(r'[+-]?\d*[.]?\d*', f.readline())
    input_size = sum(1 for _ in f) * 2
output_size = len([num for num in obj if num != ''])

# import dataset
train_dataset = AerofoilDataset(train_dir, transform=transforms.Compose([ToTensor()]))
test_dataset = AerofoilDataset(test_dir, transform=transforms.Compose([ToTensor()]))
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
        self.bn1 = nn.BatchNorm1d(num_features=hidden_layers[0])
        self.linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.bn2 = nn.BatchNorm1d(num_features=hidden_layers[1])
        self.linear3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.bn3 = nn.BatchNorm1d(num_features=hidden_layers[2])
        self.linear4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.bn4 = nn.BatchNorm1d(num_features=hidden_layers[3])
        self.linear5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.bn5 = nn.BatchNorm1d(num_features=hidden_layers[4])
        self.linear6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.bn6 = nn.BatchNorm1d(num_features=hidden_layers[5])
        self.linear7 = nn.Linear(hidden_layers[5], hidden_layers[6])
        self.bn7 = nn.BatchNorm1d(num_features=hidden_layers[6])
        self.linear8 = nn.Linear(hidden_layers[6], num_outputs)
        self.bn8 = nn.BatchNorm1d(num_features=num_outputs)
        self.relu = nn.ReLU()

        # TODO: use nn.ModuleDict to iterate through layers
        # self.nn_model = {}
        # for i, layer_size in enumerate(layers):
        #     self.nn_model[f"layer{i+1}"] = nn.Linear(layer_size, layers[i+1])
        #     if i+1 == len(layers)-1:
        #         break

    def forward(self, x):
        out = self.relu(self.bn1(self.linear1(x)))
        out = self.relu(self.bn2(self.linear2(out)))
        out = self.relu(self.bn3(self.linear3(out)))
        out = self.relu(self.bn4(self.linear4(out)))
        out = self.relu(self.bn5(self.linear5(out)))
        out = self.relu(self.bn6(self.linear6(out)))
        out = self.relu(self.bn7(self.linear7(out)))
        out = self.bn8(self.linear8(out))

        ClCd_batch = out[:, 0]
        angle_batch = out[:, 1]

        # out = x
        # for i in range(len(self.nn_model)):
        #     out = self.nn_model[f"layer{i+1}"](out)
        #     if i+1 != len(self.nn_model):  # don't do activation on last layer
        #         print("activation applied")
        #         out = self.relu(out)

        # I think MSEloss() applies an activation function to the last layer itself
        return ClCd_batch, angle_batch


model = NeuralNet(input_size, hidden_layers, output_size).to(device)

# loss and optimiser
criterion_ClCd = nn.MSELoss()
criterion_angle = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
for epoch in tqdm(range(num_epochs)):
    for i, sample in enumerate(train_loader):
        # reshape input to column vector
        sample["coordinates"] = sample["coordinates"].reshape(-1, input_size).to(device)
        ClCd_batch = sample["y"][:, 0].to(device)
        angle_batch = sample["y"][:, 1].to(device)

        # forward pass
        pred_ClCd, pred_angle = model(sample["coordinates"])
        loss_ClCd = criterion_ClCd(pred_ClCd, ClCd_batch)
        loss_angle = criterion_angle(pred_angle, angle_batch)
        loss = loss_ClCd + loss_angle

        # backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()  # do an update step to update parameters

        # print output
        if (epoch+1) % 100 == 0 and (i+1) % len(train_loader) == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}. Loss = {loss.item():.4f}")

# test set
with torch.no_grad():  # don't add gradients of test set to computational graph
    for sample_batched in test_loader:
        sample_batched["coordinates"] = sample_batched["coordinates"].reshape(-1, input_size).to(device)
        ClCd_batch = sample_batched["y"][:, 0]
        angle_batch = sample_batched["y"][:, 1]
        pred_ClCd, pred_angle = model(sample_batched["coordinates"])

        for i, (aerofoil, ClCd, angle, act_ClCd, act_angle) in enumerate(zip(sample_batched['aerofoil'], pred_ClCd,
                                                                             pred_angle, ClCd_batch, angle_batch)):
            print(f"Aerofoil {aerofoil}")
            print(f"Predictions: Max ClCd = {ClCd.item():.2f} at {angle.item():.2f}deg")
            print(f"Actual: Max ClCd = {act_ClCd.item():.2f} at {act_angle.item():.2f}deg")
            print()
