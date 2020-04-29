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
import AerofoilDataset as AD
import TitleSequence as Title
import warnings
import sys
import subprocess
if not torch.cuda.is_available():
    import ShowAerofoil as show
    from torch_lr_finder import LRFinder
    from torch.utils.tensorboard import SummaryWriter

# device configuration
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# output switches
find_LR = False  # I don't think LR is giving accurate results. A suggested LR of 1 doesn't at all work for this model
print_activations = False
print_epoch = 1  # print output after n epochs (after doing all batches within epoch)

# hyper parameters
hidden_layers = [300]
convolutions = [6, 16]
num_epochs = 35
bs = 5
learning_rate = 0.01

# file configuration
time_of_run = datetime.datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S")
path = Path(__file__).parent
train_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else 'data') / 'out' / 'train'
valid_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else 'data') / 'out' / 'valid'
test_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else 'data') / 'out' / 'test'
print_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else '') / 'print' / time_of_run
print_dir.mkdir()
if not torch.cuda.is_available():
    writer = SummaryWriter(print_dir / 'TensorBoard_events')
    TB_process = subprocess.Popen(["tensorboard", f"--logdir={path / 'print'}"], stdout=open(os.devnull, 'w'),
                                  stderr=subprocess.STDOUT)  # {print_dir} to show just this run

# title sequence
tensorboard_str = "TensorBoard running" if not torch.cuda.is_available() else "Tensorboard NOT running"
Title.print_title([" ", "Convolutional neural network", "Outputs: Max ClCd @ angle",
                   tensorboard_str, f"Output directory: {'print/' + time_of_run}"])

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

# import datasets
train_dataset = AD.AerofoilDataset(train_dir, num_channels, input_size, output_size,
                                   transform=transforms.Compose([AD.ToTensor()]))
valid_dataset = AD.AerofoilDataset(valid_dir, num_channels, input_size, output_size,
                                   transform=transforms.Compose([AD.ToTensor()]))
test_dataset = AD.AerofoilDataset(test_dir, num_channels, input_size, output_size,
                                  transform=transforms.Compose([AD.ToTensor()]))

# dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)  # bs = 1 for top_losses()

# show aerofoils
# if not torch.cuda.is_available():
    # show.show_aerofoil(writer, tensorboard=True, **train_dataset[0])
    # for i, batch in enumerate(valid_loader):
    #     show.show_aerofoil_batch(i, **batch)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        kernel_size = 10
        stride = 1  # no reduction in image size
        # dilation = 1  # default of nn.Conv1d = 1

        self.image_size = input_size
        padding = (kernel_size - 1) // stride  # true if stride=1. Produces output size = input size after filter

        self.extractor = nn.Sequential(
            nn.Conv1d(num_channels, convolutions[0], kernel_size, padding=padding, padding_mode='reflect'),
            nn.MaxPool1d(kernel_size, stride),
            nn.Conv1d(convolutions[0], convolutions[1], kernel_size, padding=padding, padding_mode='reflect'),
            nn.MaxPool1d(kernel_size, stride),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(self.image_size * convolutions[-1], hidden_layers[0]),
            nn.BatchNorm1d(num_features=num_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_layers[0], output_size),
            nn.BatchNorm1d(num_features=num_channels),
        )

    def forward(self, x):
        out = self.extractor(x)
        out = out.view(-1, num_channels, self.image_size * convolutions[-1])  # -1 for varying batch sizes
        out = self.fully_connected(out)

        ClCd = out[:, :, 0]
        angle = out[:, :, 1]

        return ClCd, angle  # must return a tensor


# model, loss and optimiser
model = ConvNet().to(device)
criterion = nn.SmoothL1Loss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=5, verbose=True)

if find_LR and not torch.cuda.is_available():
    warnings.warn("Learning rate find does NOT work! It only works when a tensor is returned from the forward method, "
                  "not a tuple.")
    if learning_rate > 0.0001:
        print(f"Selected initial learning rate too high.\nLearning rate changed to 0.0001")
        optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_finder = LRFinder(model, optimiser, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=200, num_iter=200)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    subprocess.Popen(["kill", "-9", f"{TB_process.pid}"])
    sys.exit("Learning rate plot finished")

# structure of model
# if not torch.cuda.is_available():
#     for sample in train_loader:
#         writer.add_graph(model, sample[0].float())
#     writer.close()

# training loop
model.train()  # needed?
running_train_loss = 0.
running_valid_loss = 0.
for epoch in range(num_epochs):
    for i, (train_input, train_targets, _) in enumerate(train_loader):
        # data
        train_input = train_input.to(device)  # coordinates of aerofoil(s)
        train_ClCd_target = train_targets[:, :, 0].to(device)  # max ClCd at angle
        train_angle_target = train_targets[:, :, 1].to(device)  # angle of max ClCd

        # forward pass
        train_ClCd_prediction, train_angle_prediction = model(train_input.float())
        train_loss = criterion(train_ClCd_prediction, train_ClCd_target) + criterion(train_angle_prediction,
                                                                                     train_angle_target)

        # backward pass
        optimiser.zero_grad()
        train_loss.backward()
        optimiser.step()

        if (epoch+1) % print_epoch == 0 or epoch == 0:  # at the 100th epoch:
            # calculate training loss
            running_train_loss += train_loss.item() * train_input.shape[0]   # loss.item() returns average loss per sample in batch

            # calculate validation loss
            if (i+1) % len(train_loader) == 0:  # after all batches of training set run:
                with torch.no_grad():  # don't add gradients to computational graph
                    for valid_input, valid_targets, _ in valid_loader:
                        # data
                        valid_input = valid_input.to(device)  # y coordinates of aerofoil
                        valid_ClCd_target = valid_targets[:, :, 0].to(device)  # max ClCd
                        valid_angle_target = valid_targets[:, :, 1].to(device)  # angle of max ClCd

                        # forward pass
                        valid_ClCd_prediction, valid_angle_prediction = model(valid_input.float())

                        # loss
                        running_valid_loss += (criterion(valid_ClCd_prediction, valid_ClCd_target).item() +
                                               criterion(valid_angle_prediction, valid_angle_target).item()
                                               ) * valid_input.shape[0]

                # calculate (shifted) train & validation losses (after 1 epoch)
                running_train_loss /= len(train_dataset) * 1  # average train loss (=train loss/sample)
                running_valid_loss /= len(valid_dataset) * 1

                # print to TensorBoard
                if not torch.cuda.is_available():
                    writer.add_scalar("training loss", running_train_loss, epoch)  # , epoch * len(train_dataset) + i
                    writer.add_scalar("validation loss", running_valid_loss, epoch)  # , epoch * len(train_dataset) + i

                print(f"epoch {epoch+1}/{num_epochs}, batch {i+1}/{len(train_loader)}.\n"
                      f"Training loss = {running_train_loss:.4f}, "
                      f"Validation loss = {running_valid_loss:.4f}\n")

                scheduler.step(running_valid_loss)
                running_train_loss = 0.
                running_valid_loss = 0.
if not torch.cuda.is_available():
    writer.close()
torch.save(model.state_dict(), print_dir / "model.pkl")  # create pickle file

# result = model.decode(model((valid_dataset[0])["coordinates"]).view(-1, num_channels, input_size))
# draw resulting image (in link)

# test set
model.eval()  # turn off batch normalisation and dropout
with torch.no_grad():  # don't add gradients of test set to computational graph
    losses = {}
    running_test_loss = 0.
    test_target_list = torch.tensor([]).to(device)
    test_predictions_list = torch.tensor([]).to(device)
    for test_input, test_targets, aerofoils in test_loader:
        # data
        test_coords = test_input.to(device)
        test_ClCd_target = test_targets[:, :, 0].to(device)  # max ClCd
        test_angle_target = test_targets[:, :, 1].to(device)  # angle of max ClCd

        # forward pass
        test_ClCd_prediction, test_angle_prediction = model(test_coords.float())

        # store values
        test_target_list = torch.cat((torch.cat((test_ClCd_target, test_angle_target), 1),
                                      test_target_list), 0)
        test_predictions_list = torch.cat((torch.cat((test_ClCd_prediction, test_angle_prediction), 1),
                                           test_predictions_list), 0)

        # loss
        test_loss = criterion(test_ClCd_prediction, test_ClCd_target) + criterion(test_angle_prediction,
                                                                                  test_angle_target)
        running_test_loss += test_loss.item() * test_input.shape[0]
        losses[aerofoils[0]] = test_loss.item()

    running_test_loss /= len(test_dataset) * 1  # average train loss (=train loss/sample)
    top_losses = metrics.top_losses(losses)

    print("Test set results:\n"
          f"Running test loss = {running_test_loss:.2f}\n"
          f"ClCd RMS: {metrics.root_mean_square(test_predictions_list[:, 0], test_target_list[:, 0]):.2f}, "
          f"angle RMS: {metrics.root_mean_square(test_predictions_list[:, 1], test_target_list[:, 1]):.2f}")

with open(print_dir / "test_set_results.txt", 'w') as f:
    f.write(f"Number of epochs = {num_epochs}\n"
            f"Running test loss = {running_test_loss}\n"
            f"ClCd RMS: {metrics.root_mean_square(test_predictions_list[:, 0], test_target_list[:, 0]):.2f}\n"
            f"angle RMS: {metrics.root_mean_square(test_predictions_list[:, 1], test_target_list[:, 1]):.2f}\n"
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
if not torch.cuda.is_available():
    TB_close = input("\nPress Y to kill TensorBoard...\n")
    if re.search('[yY]+', TB_close):
        subprocess.Popen(["kill", "-9", f"{TB_process.pid}"])
        print("TensorBoard closed.")
    else:
        print("TensorBorad will remain open.")
