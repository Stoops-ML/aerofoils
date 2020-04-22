import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import datetime
import ErrorMetrics as metrics
import ShowAerofoil as show
import AerofoilDataset as AD
import TitleSequence as Title
import sys
import math


# title sequence
Title.print_title([" ", "Convolutional neural network", "Outputs: Max ClCd @ angle"])

# file configuration
path = Path(__file__).parent
train_dir = path / 'data' / 'out' / 'train'
test_dir = path / 'data' / 'out' / 'test'
time_of_run = datetime.datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S")
save_model = False
print_activations = False

# device configuration
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
hidden_layers = [300, 300, 300, 300, 300, 300]
convolutions = [6, 16, 40, 120]
num_epochs = 1
bs = 100
learning_rate = 0.01  # TODO add learning rate finder

# import datasets
train_dataset = AD.AerofoilDataset(train_dir, transform=transforms.Compose([AD.ToTensor()]))
test_dataset = AD.AerofoilDataset(test_dir, transform=transforms.Compose([AD.ToTensor()]))
num_channels, input_size, output_size = AD.AerofoilDataset.get_sizes(train_dataset)

# dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=4)

# show aerofoils
# show.show_aerofoil(**train_dataset[0])
# for i, batch in enumerate(test_loader):
#     show.show_aerofoil_batch(i, **batch)


class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_outputs, convolutions):
        super(ConvNet, self).__init__()
        filter_size = 2
        stride = 2
        padding = 0

        # TODO: do batchnorms after every convolution?
        self.convolutions = nn.Sequential(
            nn.Conv1d(num_channels, convolutions[0], filter_size),  # 2 input channels (of 1D): x and y coords
            nn.MaxPool1d(2, stride),
            nn.Conv1d(convolutions[0], convolutions[1], filter_size),
            nn.MaxPool1d(2, stride),
            nn.Conv1d(convolutions[1], convolutions[2], filter_size),
            nn.MaxPool1d(2, stride),
            nn.Conv1d(convolutions[2], convolutions[3], filter_size),
            nn.MaxPool1d(2, stride))

        image_size = input_size
        for _ in range(len(convolutions)):
            calc = int(math.floor((image_size - filter_size + 2*padding) / stride + 1))
            image_size = calc
        self.image_size = calc - 1

        self.fully_connected = nn.Sequential(
            nn.Linear(self.image_size * convolutions[-1], hidden_layers[0]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.ReLU(),
            nn.Linear(hidden_layers[2], hidden_layers[3]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.ReLU(),
            nn.Linear(hidden_layers[3], hidden_layers[4]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.ReLU(),
            nn.Linear(hidden_layers[4], hidden_layers[5]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.ReLU(),
            nn.Linear(hidden_layers[5], num_outputs),
            nn.BatchNorm1d(num_features=num_channels))
        # do i have to multiply the y values of the predictions by the normalised values?

        # TODO: fix the decoder: https://stackoverflow.com/questions/55033669/encoding-and-decoding-pictures-pytorch
        # https://discuss.pytorch.org/t/visualize-feature-map/29597/6
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose1d(convolutions[3], convolutions[2], filter_size),
            # nn.ReLU(),
            nn.ConvTranspose1d(convolutions[2], convolutions[1], filter_size),
            nn.ConvTranspose1d(convolutions[1], num_channels, filter_size))

    def forward(self, x):
        out = self.convolutions(x)
        out = out.view(-1, 1, self.image_size * convolutions[-1])  # -1 for number of aerofoils in batch
        out = self.fully_connected(out)

        ClCd_batch = out[:, 0, 0]
        angle_batch = out[:, 0, 1]

        # out = x
        # for i in range(len(self.nn_model)):
        #     out = self.nn_model[f"layer{i+1}"](out)
        #     if i+1 != len(self.nn_model):  # don't do activation on last layer
        #         out = self.relu(out)

        # I think MSEloss() applies an activation function to the last layer itself
        return ClCd_batch, angle_batch

    def decode(self, x):
        return self.decoder(x)


model = ConvNet(input_size, hidden_layers, output_size, convolutions).to(device)

# loss and optimiser
criterion_ClCd = nn.MSELoss()
criterion_angle = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
model.train()  # needed?
for epoch in tqdm(range(num_epochs)):
    for i, sample in enumerate(train_loader):
        # reshape input to column vector
        sample["coordinates"] = sample["coordinates"].view(-1, 1, input_size).to(device)
        ClCd_batch = sample["y"][:, 0].to(device)
        angle_batch = sample["y"][:, 1].to(device)

        # forward pass
        pred_ClCd, pred_angle = model(sample["coordinates"].float())
        loss_ClCd = criterion_ClCd(pred_ClCd, ClCd_batch)
        loss_angle = criterion_angle(pred_angle, angle_batch)
        loss = loss_ClCd + loss_angle

        # backward pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # print output
        if (epoch+1) % 100 == 0 and (i+1) % len(train_loader) == 0:
            with torch.no_grad():  # don't add gradients of test set to computational graph
                ClCd = torch.tensor([])
                angle = torch.tensor([])
                predicted_ClCd = torch.tensor([])
                predicted_angle = torch.tensor([])
                for sample_batched in test_loader:
                    sample_batched["coordinates"] = sample_batched["coordinates"].view(-1, 1, input_size).to(device)
                    ClCd_batch = sample_batched["y"][:, 0].to(device)
                    angle_batch = sample_batched["y"][:, 1].to(device)

                    pred_ClCd, pred_angle = model(sample_batched["coordinates"].float())
                    ClCd = torch.cat((ClCd_batch, ClCd), 0)
                    angle = torch.cat((angle_batch, angle), 0)
                    predicted_ClCd = torch.cat((pred_ClCd, predicted_ClCd), 0)
                    predicted_angle = torch.cat((pred_angle, predicted_angle), 0)

            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}.\n"
                  f"Loss = {loss.item():.4f}\n"
                  f"ClCd RMS: {metrics.root_mean_square(predicted_ClCd, ClCd):.2f}, "
                  f"angle RMS: {metrics.root_mean_square(predicted_angle, angle):.2f}\n")

# result = model.decode(model((test_dataset[0])["coordinates"]).view(-1, num_channels, input_size))

# test set
model.eval()  # turn off batch normalisation and dropout
with torch.no_grad():  # don't add gradients of test set to computational graph
    ClCd = torch.tensor([])
    angle = torch.tensor([])
    predicted_ClCd = torch.tensor([])
    predicted_angle = torch.tensor([])
    for sample_batched in test_loader:
        sample_batched["coordinates"] = sample_batched["coordinates"].view(-1, 1, input_size).to(device)
        ClCd_batch = sample_batched["y"][:, 0].to(device)
        angle_batch = sample_batched["y"][:, 1].to(device)

        pred_ClCd, pred_angle = model(sample_batched["coordinates"].float())
        ClCd = torch.cat((ClCd_batch, ClCd), 0)
        angle = torch.cat((angle_batch, angle), 0)
        predicted_ClCd = torch.cat((pred_ClCd, predicted_ClCd), 0)
        predicted_angle = torch.cat((pred_angle, predicted_angle), 0)

print("Results after training:\n"
      f"ClCd RMS: {metrics.root_mean_square(predicted_ClCd, ClCd):.2f}, "
      f"angle RMS: {metrics.root_mean_square(predicted_angle, angle):.2f}")

if save_model:
    print_dir = path / 'print' / time_of_run
    print_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), print_dir / "model.pkl")  # creates pickle file
    with open(print_dir / "RESULTS.txt", 'w') as f:
        f.write(f"Number of epochs = {num_epochs}\n"
            f"ClCd: RMS = {metrics.root_mean_square(predicted_ClCd, ClCd):.2f}, "
            f"R2 = {metrics.R2_score(predicted_ClCd, ClCd):.2f}\n"
            f"angle: RMS = {metrics.root_mean_square(predicted_angle, angle):.2f}, "
            f"R2 = {metrics.R2_score(predicted_angle, angle):.2f}\n")

# Visualize feature maps
if print_activations:
    activation = {}
    def get_activation(name):
        def hook(_, __, output):
            activation[name] = output.detach()
        return hook

    model.conv2.register_forward_hook(get_activation('conv2'))
    output = model((test_dataset[0])["coordinates"].view(1, num_channels, input_size).float().to(device))

    act = activation['conv2'].squeeze()
    # fig, axarr = plt.subplots(act.size(0))
    plt.plot((test_dataset[0])["coordinates"].view(-1, input_size).numpy()[0], 'r+')
    for idx in range(act.size(0)):
        # axarr[idx].plot(act[idx])
        plt.plot(act[idx])
    plt.show()
