import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
import numpy as np
import datetime
import ErrorMetrics as metrics
import ShowAerofoil as show
import AerofoilDataset as AD
import TitleSequence as Title
import warnings
from torch.utils.tensorboard import SummaryWriter
import sys
import subprocess
from torch_lr_finder import LRFinder


# output switches
find_LR = False
print_activations = False
print_epoch = 1  # print output after n epochs (after doing all batches within epoch)

# hyper parameters
hidden_layers = [300]
convolutions = [6]  # , 16, 40, 120]
num_epochs = 2
bs = 25
learning_rate = 2

# file configuration
time_of_run = datetime.datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S")
path = Path(__file__).parent
train_dir = path / 'data' / 'out' / 'train'
valid_dir = path / 'data' / 'out' / 'valid'
test_dir = path / 'data' / 'out' / 'test'
print_dir = path / 'print' / time_of_run
writer = SummaryWriter(print_dir / 'TensorBoard_events')
TB_process = subprocess.Popen(["tensorboard", f"--logdir={path / 'print'}"], stdout=open(os.devnull, 'w'),
                              stderr=subprocess.STDOUT)  # {print_dir} to show just this run

# write log file of architecture and for note taking
with open(print_dir / "log.txt", 'w') as f:
    f.write(re.sub(r'_', '/', time_of_run) + "\n")
    f.write(f"Epochs: {num_epochs}\n")
    f.write(f"Learning rate = {learning_rate}\n")
    f.write(f"Number of layers = {len(hidden_layers) + 2}\n")
    f.write(f"Hidden layers = {hidden_layers}\n")
    f.write(f"Number of convolutions = {len(convolutions)}\n")
    f.write(f"Convolutions = {convolutions}\n")

# get input size, output size and number of channels
file = os.listdir(train_dir)[0] if '.csv' in os.listdir(train_dir)[0] else os.listdir(train_dir)[1]  # sample file
coords = np.loadtxt(train_dir / file, delimiter=" ", dtype=np.float32, skiprows=1)  # xy coordinates of sample
input_size = len(coords)
with open(train_dir / file) as f:
    line = f.readline()
    y_vals = [float(num) for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']  # outputs of sample
    output_size = len(y_vals)
num_channels = 1  # one channel for y coordinate (xy coordinates requires two channels)

# title sequence
Title.print_title([" ", "Convolutional neural network", "Outputs: Max ClCd @ angle",
                   "TensorBoardX running", f"Output directory: {'print/' + time_of_run}"])

# device configuration
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import datasets
train_dataset = AD.AerofoilDataset(train_dir, num_channels, input_size, output_size,
                                   transform=transforms.Compose([AD.ToTensor()]))
valid_dataset = AD.AerofoilDataset(valid_dir, num_channels, input_size, output_size,
                                   transform=transforms.Compose([AD.ToTensor()]))
test_dataset = AD.AerofoilDataset(test_dir, num_channels, input_size, output_size,
                                  transform=transforms.Compose([AD.ToTensor()]))

# dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)  # bs = 1 for top_losses()

# show aerofoils
# show.show_aerofoil(writer, tensorboard=True, **train_dataset[0])
# for i, batch in enumerate(valid_loader):
#     show.show_aerofoil_batch(i, **batch)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kernel_size = 2
        stride = 2
        dilation = 1  # default of nn.Conv1d = 1
        padding = ((input_size - 1) * stride - input_size + kernel_size + (kernel_size - 1) * (dilation - 1)
                   ) // 2
        self.image_size = (input_size + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1))\
                          // stride + 1  # image size according to padding (as padding might not actually be an int)

        self.extractor = nn.Sequential(
            nn.Conv1d(num_channels, convolutions[0], kernel_size, padding=padding, padding_mode='reflect'),  # 2 input channels (of 1D): x and y coords
            nn.MaxPool1d(2, stride),
            # nn.Conv1d(convolutions[0], convolutions[1], kernel_size, padding=padding, padding_mode='reflect'),
            # nn.MaxPool1d(2, stride),
            # nn.Conv1d(convolutions[1], convolutions[2], kernel_size, padding=padding, padding_mode='reflect'),
            # nn.MaxPool1d(2, stride),
            # nn.Conv1d(convolutions[2], convolutions[3], kernel_size, padding=padding, padding_mode='reflect'),
            # nn.MaxPool1d(2, stride)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(self.image_size * convolutions[-1], hidden_layers[0]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], output_size),
            nn.BatchNorm1d(num_features=num_channels))

        # TODO: fix the decoder: https://stackoverflow.com/questions/55033669/encoding-and-decoding-pictures-pytorch
        # https://discuss.pytorch.org/t/visualize-feature-map/29597/6
        # self.decoder = torch.nn.Sequential(
            # nn.ConvTranspose1d(convolutions[3], convolutions[2], filter_size),
            # nn.ReLU(),
            # nn.ConvTranspose1d(convolutions[2], convolutions[1], filter_size),
            # nn.ConvTranspose1d(convolutions[1], num_channels, filter_size))

    def forward(self, x):
        out = self.extractor(x)
        out = out.view(-1, num_channels, self.image_size * convolutions[-1])  # -1 for number of aerofoils in batch
        out = self.fully_connected(out)

        return out  # you don't want to return a tuple, you must return a tensor

    def decode(self, x):
        return self.decoder(x)


# model, loss and optimiser
model = ConvNet().to(device)
criterion = nn.SmoothL1Loss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=5, verbose=True)

if find_LR:
    if learning_rate > 0.0001:
        warnings.warn(f"Starting learning should be lowered from {learning_rate}")
    lr_finder = LRFinder(model, optimiser, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    sys.exit("Learning rate plot finished")

# structure of model
for sample in train_loader:
    writer.add_graph(model, sample[0].float())
writer.close()

# training loop
model.train()  # needed?
running_train_loss = 0.
running_valid_loss = 0.
for epoch in range(num_epochs):
    for i, train_sample in enumerate(train_loader):
        # data
        train_input = train_sample[0].to(device)  # coordinates of aerofoil(s)
        train_targets = train_sample[1].to(device)  # max ClCd at angle

        # forward pass
        train_predictions = model(train_input.float())
        train_loss = criterion(train_predictions, train_targets)

        # backward pass
        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

        if (epoch+1) % print_epoch == 0:  # at the 100th epoch:
            # calculate training loss
            running_train_loss += train_loss.item() * len(train_sample)   # loss.item() returns average loss per sample in batch

            # calculate validation loss
            if (i+1) % len(train_loader) == 0:  # after all batches of training set run:
                with torch.no_grad():  # don't add gradients to computational graph
                    for valid_sample in valid_loader:
                        # data
                        valid_input = valid_sample[0].to(device)  # y coordinates of aerofoil
                        valid_targets = valid_sample[1].to(device)  # max ClCd at angle

                        # forward pass
                        valid_predictions = model(valid_input.float())

                        # loss
                        running_valid_loss += criterion(valid_predictions, valid_targets).item() * len(valid_sample)

                # calculate (shifted) train & validation losses (after 1 epoch)
                running_train_loss /= len(train_dataset) * 1  # average train loss (=train loss/sample)
                running_valid_loss /= len(valid_dataset) * 1

                # print to TensorBoard
                writer.add_scalar("training loss", running_train_loss, epoch)  # , epoch * len(train_dataset) + i
                writer.add_scalar("validation loss", running_valid_loss, epoch)  # , epoch * len(train_dataset) + i

                print(f"epoch {epoch+1}/{num_epochs}, batch {i+1}/{len(train_loader)}.\n"
                      f"Training loss = {running_train_loss:.4f}, "
                      f"Validation loss = {running_valid_loss:.4f}\n")

                scheduler.step(running_valid_loss)
                running_train_loss = 0.
                running_valid_loss = 0.
writer.close()
torch.save(model.state_dict(), print_dir / "model.pkl")  # create pickle file

# result = model.decode(model((valid_dataset[0])["coordinates"]).view(-1, num_channels, input_size))
# draw resulting image (in link)

# test set
model.eval()  # turn off batch normalisation and dropout
with torch.no_grad():  # don't add gradients of test set to computational graph
    losses = {}
    running_test_loss = 0.
    test_target_list = torch.tensor([])
    test_predictions_list = torch.tensor([])
    for test_batch in test_loader:
        # data
        test_coords = test_batch[0].to(device)
        test_targets = test_batch[1].to(device)  # max ClCd at angle

        # forward pass
        test_predictions = model(test_coords.float())
        test_target_list = torch.cat((test_targets, test_target_list), 0)
        test_predictions_list = torch.cat((test_predictions, test_predictions_list), 0)

        # loss
        test_loss = criterion(test_predictions, test_targets)
        running_test_loss += test_loss.item() * len(test_batch)
        losses[test_batch[2][0]] = test_loss.item()

    running_test_loss /= len(test_dataset) * 1  # average train loss (=train loss/sample)
    top_losses = metrics.top_losses(losses)

    print("Test set results:\n"
          f"Running test loss = {running_test_loss:.2f}\n"
          f"ClCd RMS: {metrics.root_mean_square(test_predictions_list[:, :, 0], test_target_list[:, :, 0]):.2f}, "
          f"angle RMS: {metrics.root_mean_square(test_predictions_list[:, :, 1], test_target_list[:, :, 1]):.2f}")

with open(print_dir / "test_set_results.txt", 'w') as f:
    f.write(f"Number of epochs = {num_epochs}\n"
            f"Running test loss = {running_test_loss}\n"
            f"ClCd RMS: {metrics.root_mean_square(test_predictions_list[:, :, 0], test_target_list[:, :, 0]):.2f}\n"
            f"angle RMS: {metrics.root_mean_square(test_predictions_list[:, :, 1], test_target_list[:, :, 1]):.2f}\n"
            f"\nTop losses:\n")

    for i, (k, v) in enumerate(top_losses.items()):
        f.write(f"{i}. {k}: {v:.2f}\n")

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
TB_close = input("\nPress Y to kill TensorBoard...\n")
if re.search('[yY]+', TB_close):
    subprocess.Popen(["kill", "-9", f"{TB_process.pid}"])
    print("TensorBoard closed.")
else:
    print("TensorBorad will remain open.")
