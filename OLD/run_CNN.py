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
from OLD import TitleSequence as Title
from NeuralNets import *
import subprocess
if not torch.cuda.is_available():
    from torch_lr_finder import LRFinder
    import torch.utils.tensorboard as tf


# output switches
find_LR = False
print_activations = False
print_heatmap = False
print_comp_graph = True
print_epoch = 1  # calculate and print output & plot losses after n epochs (after doing all batches within epoch)
if torch.cuda.is_available():  # not available on cuda
    find_LR = False
    print_activations = False
    print_heatmap = False
    print_comp_graph = False

# hyper parameters
hidden_layers = [50]
# convolutions = [6, 16, 32]  for ConvNet
convolutions = [64, 46, 46, 30]  # input/output channels of convolutions
num_epochs = 1
bs = 5
learning_rate = 0.01

# paths and files
time_of_run = datetime.datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S")
path = Path(__file__).parent
train_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else 'data') / 'out' / 'train'
valid_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else 'data') / 'out' / 'valid'
test_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else 'data') / 'out' / 'test'
print_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else '') / 'print' / time_of_run
print_dir.mkdir()

# TensorBoard writer
if not torch.cuda.is_available():
    writer = tf.SummaryWriter(print_dir / 'TensorBoard_events')
    TB_process = subprocess.Popen(["tensorboard", f"--logdir={print_dir.parent}"], stdout=open(os.devnull, 'w'),
                                  stderr=subprocess.STDOUT)  # {print_dir} to show just this run

# device configuration
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# title sequence
tensorboard_str = "TensorBoard running" if not torch.cuda.is_available() else "Tensorboard NOT running"
Title.print_title([" ", "Densely connected CNN", "Outputs: Max ClCd @ angle",
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
coords = np.loadtxt(train_dir / os.listdir(train_dir)[0], delimiter=" ", dtype=np.float32, skiprows=1)  # coords of sample
input_size = len(coords)
with open(train_dir / os.listdir(train_dir)[0]) as f:
    line = f.readline()  # max ClCd & angle are on first line of file
    y_vals = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']  # outputs of sample
    output_size = len(y_vals)
num_channels = 1  # one channel for y coordinate (xy coordinates requires two channels)

# import datasets. Use same scaling for train, valid and test sets
train_dataset = AD.AerofoilDataset(train_dir, num_channels, input_size, output_size,
                                   transform=transforms.Compose([AD.NormaliseYValues(train_dir)]))
valid_dataset = AD.AerofoilDataset(valid_dir, num_channels, input_size, output_size,
                                   transform=transforms.Compose([AD.NormaliseYValues(train_dir)]))
test_dataset = AD.AerofoilDataset(test_dir, num_channels, input_size, output_size,
                                  transform=transforms.Compose([AD.NormaliseYValues(train_dir)]))

# dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=4)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs*2, shuffle=False, num_workers=4)  # not storing gradients, so can double bs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)  # bs=1 required for top_losses()

# model, loss and optimiser
model = DenseNet(input_size, output_size, convolutions, hidden_layers, num_channels, 32).to(device)
print(model)
# model = ConvNet(input_size, convolutions, num_channels, hidden_layers, output_size).to(device)
criterion = metrics.MyLossFunc()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=5, verbose=True)

# find learning rate
if find_LR and not torch.cuda.is_available():
    if learning_rate > 0.0001:
        print(f"Selected initial learning rate too high.\nLearning rate changed to 0.0001")
        optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_finder = LRFinder(model, optimiser, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=200, num_iter=200)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    subprocess.Popen(["kill", "-9", f"{TB_process.pid}"])
    sys.exit("Learning rate plot finished")

# computational graph
if print_comp_graph:
    for sample in train_loader:
        writer.add_graph(model, sample[0].float())
    writer.close()

# training loop
running_train_loss = 0.
running_valid_loss = 0.
for epoch in range(num_epochs):
    for i, (train_input, train_targets, _) in enumerate(train_loader):
        # data
        train_input = train_input.to(device)  # coordinates of aerofoil(s)
        train_targets = train_targets.to(device)  # max ClCd at angle

        # forward pass
        train_predictions = model(train_input.float())
        train_loss = criterion(train_predictions, train_targets)  # matches LRFinder()

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
                    model.eval()
                    for valid_input, valid_targets, _ in valid_loader:

                        # data
                        valid_input = valid_input.to(device)  # y coordinates of aerofoil
                        valid_targets = valid_targets.to(device)  # max ClCd at angle

                        # forward pass
                        valid_predictions = model(valid_input.float())
                        running_valid_loss += criterion(valid_predictions, valid_targets).item() * valid_input.shape[0]

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
                model.train()

if not torch.cuda.is_available():
    writer.close()
torch.save(model.state_dict(), print_dir / "model.pkl")  # create pickle file

# test set
model.eval()  # turn off batch normalisation and dropout
with torch.no_grad():  # don't add gradients of test set to computational graph
    losses = {}
    running_test_loss = 0.
    test_targets_list = torch.tensor([]).to(device)
    test_predictions_list = torch.tensor([]).to(device)
    for test_input, test_targets, aerofoils in test_loader:
        # data
        test_coords = test_input.to(device)
        test_targets = test_targets.to(device)  # max ClCd at angle

        # forward pass
        test_predictions = model(test_coords.float())

        # store values
        # test_targets three dimensional here because you have a batch (1 dimension), and two more dimensions from
        # the .view(1, self.output_size) on the self.y[item] variable in the dataset class AerofoilDataset.py.
        # Need a 2D tensor: one for ClCd and one for angle.
        test_targets_list = torch.cat((test_targets_list, test_targets[0]), 0)
        test_predictions_list = torch.cat((test_predictions_list,
                                           torch.tensor(test_predictions).view(1, output_size).to(device)), 0)

        # loss
        test_loss = criterion(test_predictions, test_targets)  # matches LRFinder()
        running_test_loss += test_loss.item() * test_input.shape[0]
        losses[aerofoils[0]] = test_loss.item()  # this requires batchsize = 1

    running_test_loss /= len(test_dataset) * 1  # average train loss (=train loss/sample)
    top_losses = metrics.top_losses(losses)

    print("Test set results:\n"
          f"Running test loss = {running_test_loss:.4f}\n"
          f"ClCd RMS: {metrics.root_mean_square(test_predictions_list[:, 0], test_targets_list[:, 0]):.2f}, "
          f"angle RMS: {metrics.root_mean_square(test_predictions_list[:, 1], test_targets_list[:, 1]):.2f}")

with open(print_dir / "test_set_results.txt", 'w') as f:
    f.write(f"Number of epochs = {num_epochs}\n"
            f"Running test loss = {running_test_loss:.4f}\n"
            f"ClCd RMS: {metrics.root_mean_square(test_predictions_list[:, 0], test_targets_list[:, 0]):.2f}\n"
            f"angle RMS: {metrics.root_mean_square(test_predictions_list[:, 1], test_targets_list[:, 1]):.2f}\n"
            f"\nTop losses:\n")

    for i, (k, v) in enumerate(top_losses.items()):
        f.write(f"{i}. {k}: {v:.2f}\n")

if print_activations:
    def get_activation(name):
        def hook(_, __, output):
            activation[name] = output.detach()
        return hook

    # use this to find whether activations are useful (i.e. whether they have a value - if 0 then they're not useful)
    for i in range(len(convolutions)):
        # initialise hook
        activation = {}
        activation_str = f'Convolution_#{i}'
        model.block_and_layer[i * 4].register_forward_hook(get_activation(activation_str))  # todo fix this

        # get activations
        x, y, aerofoil = next(iter(test_loader))  # using test_loader as it has a batchsize of 1
        output = model(x.float().to(device))
        act = activation[activation_str].squeeze()

        # plot activations
        fig, axarr = plt.subplots(act.size(0))
        for idx in range(act.size(0)):
            axarr[idx].plot(act[idx])
        fig.suptitle(f"Activations of {activation_str}\nAerofoil {aerofoil[0]}")
        plt.show()
        writer.add_figure(f"Activations of {activation_str}\nAerofoil {aerofoil[0]}", fig, global_step=num_epochs)
        writer.close()  # todo TB doesn't print image

if print_heatmap:
    def get_activation(name):
        def hook(_, __, output):
            activation[name] = output.detach()
        return hook

    # initialise hook
    activation = {}
    model.block_and_layer[-1].register_forward_hook(get_activation('Last_layer'))

    # get activations
    x, y, aerofoil = next(iter(test_loader))  # using test_loader as it has a batchsize of 1
    output = model(x.float().to(device))
    act = activation['Last_layer'].squeeze()

    # heat map of last layer
    average_act = 0
    for channel in act:
        average_act += channel
    average_act /= act.size(0)
    fig, ax = plt.subplots()
    ax.plot(x.squeeze())
    im = ax.imshow(average_act.view(1, -1), alpha=0.5, extent=(0, input_size, torch.min(x) * 1.5, torch.max(x) * 1.5),
                   interpolation='bilinear', cmap='magma', aspect=input_size)
    # fig.suptitle(f"Heat map of last_layer\nAerofoil {aerofoil[0]}")
    # plt.show()
    writer.add_figure(f"Heat map of last_layer\nAerofoil {aerofoil[0]}", fig, global_step=num_epochs)
    writer.close()  # todo figure not printing to TB (but prints to plt.show)

# kill TensorBoard
if not torch.cuda.is_available():
    TB_close = input("\nPress Y to kill TensorBoard...\n")
    if re.search('[yY]+', TB_close):
        subprocess.Popen(["kill", "-9", f"{TB_process.pid}"])
        print("TensorBoard closed.")
    else:
        print("TensorBorad will remain open.")
