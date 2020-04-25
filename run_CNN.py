import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import re
import datetime
import ErrorMetrics as metrics
import ShowAerofoil as show
import AerofoilDataset as AD
import TitleSequence as Title
from torch.utils.tensorboard import SummaryWriter
import sys
import math
import subprocess

# output switches
print_activations = False
run_tensorboard = True
print_epoch = 100  # print output after n epochs (after doing all batches within epoch)

# hyper parameters
hidden_layers = [300, 300, 300, 300, 300, 300]
convolutions = [6, 16, 40, 120]
num_epochs = 1
bs = 25
learning_rate = 0.2  # TODO add learning rate finder

# file configuration
time_of_run = datetime.datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S")
path = Path(__file__).parent
train_dir = path / 'data' / 'out' / 'train'
valid_dir = path / 'data' / 'out' / 'valid'
test_dir = path / 'data' / 'out' / 'test'
print_dir = path / 'print' / time_of_run
writer = SummaryWriter(print_dir / 'TensorBoard_events')
if run_tensorboard:
    TB_process = subprocess.Popen(["tensorboard", f"--logdir={path / 'print'}"],  # {print_dir} to show just this run
                                  stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)

# write log file of architecture and for note taking
with open(print_dir / "log.txt", 'w') as f:
    f.write(re.sub(r'_', '/', time_of_run) + "\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Learning rate = {learning_rate}\n")
    f.write(f"Number of layers = {len(hidden_layers) + 2}\n")
    f.write(f"Hidden layers = {hidden_layers}\n")
    f.write(f"Number of convolutions = {len(convolutions)}\n")
    f.write(f"Convolutions = {convolutions}\n")

# title sequence
run_tensorboard_str = "TENSORBOARD RUNNING" if run_tensorboard else "TensorBoard not running"
Title.print_title([" ", "Convolutional neural network", "Outputs: Max ClCd @ angle",
                   f"{run_tensorboard_str}", f"Output directory: {'print/' + time_of_run}"])

# device configuration
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import datasets
train_dataset = AD.AerofoilDataset(train_dir, transform=transforms.Compose([AD.ToTensor()]))
valid_dataset = AD.AerofoilDataset(valid_dir, transform=transforms.Compose([AD.ToTensor()]))
test_dataset = AD.AerofoilDataset(test_dir, transform=transforms.Compose([AD.ToTensor()]))
num_channels, input_size, output_size = AD.AerofoilDataset.get_sizes(train_dataset)

# dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)  # bs = 1 for top_losses()

# show aerofoils
# show.show_aerofoil(writer, tensorboard=True, **train_dataset[0])
# for i, batch in enumerate(valid_loader):
#     show.show_aerofoil_batch(i, **batch)


class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_outputs, convolutions):
        super(ConvNet, self).__init__()
        filter_size = 2
        stride = 2
        padding = 0

        # TODO: normalise the y values
        # TODO: do batchnorms after every convolution?
        # TODO: too much pooling?
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
            nn.BatchNorm1d(num_features=num_channels)
            )

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
criterion_ClCd = nn.SmoothL1Loss()
criterion_angle = nn.SmoothL1Loss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

# structure of model
for sample in train_loader:
    writer.add_graph(model, sample["coordinates"].view(-1, 1, input_size).float())
writer.close()

# training loop
model.train()  # needed?
train_loss = 0.
valid_loss = 0.
for epoch in (range(num_epochs)):  # tqdm doesn't seem to work with nested loops with one progress bar
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

        if (epoch+1) % print_epoch == 0:  # at the 100th epoch:
            # calculate training loss
            train_loss += loss.item() * len(sample)   # loss.item() returns average loss per sample in batch

            # calculate validation loss
            if (i+1) % len(train_loader) == 0:  # after all batches of training set run:
                # TODO make better variable names here
                with torch.no_grad():  # don't add gradients of test set to computational graph
                    ClCd = torch.tensor([])
                    angle = torch.tensor([])
                    predicted_ClCd = torch.tensor([])
                    predicted_angle = torch.tensor([])
                    for sample_batched in valid_loader:
                        sample_batched["coordinates"] = sample_batched["coordinates"].view(-1, 1, input_size).to(device)
                        ClCd_batch = sample_batched["y"][:, 0].to(device)
                        angle_batch = sample_batched["y"][:, 1].to(device)

                        pred_ClCd, pred_angle = model(sample_batched["coordinates"].float())
                        ClCd = torch.cat((ClCd_batch, ClCd), 0)
                        angle = torch.cat((angle_batch, angle), 0)
                        predicted_ClCd = torch.cat((pred_ClCd, predicted_ClCd), 0)
                        predicted_angle = torch.cat((pred_angle, predicted_angle), 0)

                    valid_loss += (criterion_angle(pred_angle, angle_batch) +
                                   criterion_ClCd(pred_ClCd, ClCd_batch)).item() * len(sample_batched)

                # print output to tensorboard and screen
                train_loss /= len(train_dataset) * 1  # average train loss (=train loss/sample)
                valid_loss /= len(valid_dataset) * 1  # losses calculated after 1 epoch

                writer.add_scalar("training loss", train_loss, epoch)  # , epoch * len(train_dataset) + i
                writer.add_scalar("validation loss", valid_loss, epoch)  # , epoch * len(train_dataset) + i

                print(f"epoch {epoch+1}/{num_epochs}, batch {i+1}/{len(train_loader)}.\n"
                      f"Training loss = {train_loss:.4f}, "
                      f"Validation loss = {valid_loss:.4f}\n"
                      f"ClCd RMS: {metrics.root_mean_square(predicted_ClCd, ClCd):.2f}, "
                      f"angle RMS: {metrics.root_mean_square(predicted_angle, angle):.2f}\n")

                train_loss = 0.
                valid_loss = 0.
writer.close()
torch.save(model.state_dict(), print_dir / "model.pkl")  # create pickle file

# result = model.decode(model((valid_dataset[0])["coordinates"]).view(-1, num_channels, input_size))
# draw resulting image (in link)

# test set
# note this is actually using the validation dataset
model.eval()  # turn off batch normalisation and dropout
losses = {}
with torch.no_grad():  # don't add gradients of test set to computational graph
    ClCd = torch.tensor([])
    angle = torch.tensor([])
    predicted_ClCd = torch.tensor([])
    predicted_angle = torch.tensor([])
    for sample_batched in test_loader:
        sample_batched["coordinates"] = sample_batched["coordinates"].view(-1, 1, input_size).to(device)
        target_ClCd = sample_batched["y"][:, 0].to(device)
        target_angle = sample_batched["y"][:, 1].to(device)

        pred_ClCd, pred_angle = model(sample_batched["coordinates"].float())
        ClCd = torch.cat((target_ClCd, ClCd), 0)
        angle = torch.cat((target_angle, angle), 0)
        predicted_ClCd = torch.cat((pred_ClCd, predicted_ClCd), 0)
        predicted_angle = torch.cat((pred_angle, predicted_angle), 0)

        # get losses
        loss_ClCd = criterion_ClCd(pred_ClCd, target_ClCd)
        loss_angle = criterion_angle(pred_angle, target_angle)
        loss = loss_ClCd + loss_angle
        losses[sample_batched["aerofoil"][0]] = loss.item()

        with open(print_dir / "test_set_results.txt", 'a') as f:
            for i, (pred_angle, pred_ClCd, y_angle, y_ClCd, aerofoil) in enumerate(zip(pred_angle, pred_ClCd, target_angle, target_ClCd, sample_batched["aerofoil"])):
                f.write(f"{i + 1}. {aerofoil}:\n"
                        f"predicted angle = {pred_angle:.2f}, target angle = {y_angle:.2f}\n"
                        f"predicted ClCd = {pred_ClCd:.2f}, target ClCd = {y_ClCd:.2f}\n\n")

    top_losses = metrics.top_losses(losses)
    print(top_losses)
    sys.exit()

    print("Test set results:\n"
          f"ClCd RMS: {metrics.root_mean_square(predicted_ClCd, ClCd):.2f}, "
          f"angle RMS: {metrics.root_mean_square(predicted_angle, angle):.2f}")

with open(print_dir / "test_set_results.txt", 'a') as f:
    f.write(f"\nNumber of epochs = {num_epochs}\n"
            f"ClCd: overall RMS = {metrics.root_mean_square(predicted_ClCd, ClCd):.2f}, "
            f"overall R2 = {metrics.R2_score(predicted_ClCd, ClCd):.2f}\n"
            f"angle: overall RMS = {metrics.root_mean_square(predicted_angle, angle):.2f}, "
            f"overall R2 = {metrics.R2_score(predicted_angle, angle):.2f}\n")

# Visualize feature maps
if print_activations:
    # TODO: add this to tensorboard
    activation = {}
    def get_activation(name):
        def hook(_, __, output):
            activation[name] = output.detach()
        return hook

    # model.conv2.register_forward_hook(get_activation('conv2'))
    model.convolutions[6].register_forward_hook(get_activation('conv6'))
    sample = (valid_dataset[0])["coordinates"]
    output = model(sample.view(1, num_channels, input_size).float().to(device))

    act = activation['conv6'].squeeze()
    fig, axarr = plt.subplots(act.size(0))

    for idx in range(act.size(0)):
        axarr[idx].plot(act[idx])
        plt.plot(act[idx])

# kill TensorBoard
if run_tensorboard:
    TB_close = input("\nPress Y to kill TensorBoard...\n")
    if re.search('[yY]+', TB_close):
        subprocess.Popen(["kill", "-9", f"{TB_process.pid}"])
        print("TensorBoard closed.")
    else:
        print("TensorBorad will remain open.")
