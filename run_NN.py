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
import sys
import AerofoilDataset as AD
import TitleSequence as Title
import ShowAerofoil as show


# title sequence
Title.print_title([" ", "Neural network", "Outputs: Max ClCd @ angle"])

# file configuration
path = Path(__file__).parent
train_dir = path / 'data' / 'out' / 'train'
test_dir = path / 'data' / 'out' / 'test'
print_dir = path / 'print'
print_dir.mkdir(exist_ok=True)
time_of_run = datetime.datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S")

# device configuration
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
hidden_layers = [300, 300, 300, 300, 300, 300, 300, 300]
num_epochs = 400
bs = 50
learning_rate = 0.01  # TODO add learning rate finder

# import dataset
train_dataset = AD.AerofoilDataset(train_dir, transform=transforms.Compose([AD.ToTensor()]))
test_dataset = AD.AerofoilDataset(test_dir, transform=transforms.Compose([AD.ToTensor()]))
num_channels, input_size, output_size = AD.AerofoilDataset.get_sizes(train_dataset)
input_size *= num_channels  # dataset needs to be flattened

# dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=4)

# show aerofoils
# show.show_aerofoil(**train_dataset[0])
# for i, batch in enumerate(test_loader):
#     show.show_aerofoil_batch(i, **batch)


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

        # TODO: use nn.ModuleDict to iterate through layers
        # self.nn_model = {}
        # for i, layer_size in enumerate(layers):
        #     self.nn_model[f"layer{i+1}"] = nn.Linear(layer_size, layers[i+1])
        #     if i+1 == len(layers)-1:
        #         break

    def forward(self, x):
        out = F.relu(self.bn1(self.linear1(x)))
        out = F.relu(self.bn2(self.linear2(out)))
        out = F.relu(self.bn3(self.linear3(out)))
        out = F.relu(self.bn4(self.linear4(out)))
        out = F.relu(self.bn5(self.linear5(out)))
        out = F.relu(self.bn6(self.linear6(out)))
        out = F.relu(self.bn7(self.linear7(out)))
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
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
model.train()  # needed?
for epoch in tqdm(range(num_epochs)):
    for i, sample in enumerate(train_loader):
        # reshape input to column vector
        sample["coordinates"] = sample["coordinates"].view(-1, input_size).to(device)
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
        optimiser.step()  # do an update step to update parameters

        # print output
        if (epoch+1) % 100 == 0 and (i+1) % len(train_loader) == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{len(train_loader)}. Loss = {loss.item():.4f}")

torch.save(model.state_dict(), print_dir / (time_of_run + ".pkl"))  # creates pickle file


# test set
# TODO add validation set to epoch to calculate...
# loaded_model = NeuralNet(input_size, hidden_layers, output_size).to(device)  # same as trained model
# loaded_model.load_state_dict(torch.load(FILE, map_location=device))  # load trained model
# be careful of what device of map_location is. It changes if model trained on CPU or GPU and loading onto a CPU or GPU
# ^see Python Engineer tutorial on this
# loaded_model.to(device)
# loaded_model.eval()
model.eval()  # turn off batch normalisation and dropout
with torch.no_grad():  # don't add gradients of test set to computational graph
    for sample_batched in test_loader:
        sample_batched["coordinates"] = sample_batched["coordinates"].view(-1, input_size).to(device)
        ClCd_batch = sample_batched["y"][:, 0].to(device)
        angle_batch = sample_batched["y"][:, 1].to(device)
        pred_ClCd, pred_angle = model(sample_batched["coordinates"].float())
        print(f"ClCd RMS: {metrics.root_mean_square(pred_ClCd, ClCd_batch):.2f}")
        print(f"angle RMS: {metrics.root_mean_square(pred_angle, angle_batch):.2f}")

        # with open(print_dir / test_out, 'w') as f:
        #     spacing = 13
        #     f.write(f"{'Aerofoil':<{spacing}}"
        #             f"{'Pred_ClCd':^{spacing}}{'Targ_ClCd':^{spacing}}{'RMS':^{spacing}}"
        #             f"{'Pred_angle':^{spacing}}{'Targ_angle':^{spacing}}{'RMS':^{spacing}}\n")

            # for i, (aerofoil, ClCd, angle, act_ClCd, act_angle) in enumerate(zip(sample_batched['aerofoil'], pred_ClCd,
            #                                                                      pred_angle, ClCd_batch, angle_batch)):

                # # print file
                # f.write(f"{aerofoil[:-4]:<{spacing}}"
                #         f"{ClCd.item():^{spacing}.2f}{act_ClCd.item():^{spacing}.2f}"
                #         f"{angle.item():^{spacing}.2f}{act_angle.item():^{spacing}.2f}\n")


