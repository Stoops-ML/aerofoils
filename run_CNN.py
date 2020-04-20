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
import sys


# title sequence
strings2print = [f" ",
                 # f"Weight files to run: {model_names}",
                 f"Outputs: Max ClCd @ angle"]
spacing = 60
print("#"*spacing)
print(f"#{'Aerofoil regression problem':^{spacing-2}}#")
for string in strings2print:
    print(f"# {string:<{spacing-4}} #")
print("#"*spacing)
print()


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
        x = []
        y = []
        with open(self.root_dir / self.aerofoils[item]) as f:
            for line in f:
                if re.search(r'ClCd', line):  # find y values
                    y_vals = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
                    max_ClCd, angle = float(y_vals[0]), float(y_vals[1])
                    continue
                xy = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']
                x.append(float(xy[0]))
                y.append(float(xy[1]))
        coords = np.stack((x, y), axis=0)  # perhaps using np.loadtxt() is better

        sample = {"aerofoil": self.aerofoils[item], "coordinates": coords, "y": [max_ClCd, angle]}

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
train_dir = path / 'data' / 'out' / 'train'
test_dir = path / 'data' / 'out' / 'test'
time_of_run = datetime.datetime.now().strftime("D%d_%m_%Y_T%H_%M_%S")
print_dir = path / 'print' / time_of_run
print_dir.mkdir(exist_ok=True)

# device configuration
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
hidden_layers = [300, 300, 300, 300, 300, 300, 300, 300]
num_epochs = 5000
bs = 50
learning_rate = 0.01  # TODO add learning rate finder

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


class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_outputs):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(2, 6, 2)  # 2 input channels (of 1D): x and y. 6 output channels
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(6, 16, 2)
        self.fc1 = nn.Linear(464, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 84)
        self.fc5 = nn.Linear(84, 50)
        self.fc6 = nn.Linear(50, 50)
        self.fc7 = nn.Linear(50, 50)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        out = out.view(-1, 1, 464)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        out = self.fc7(out)

        ClCd_batch = out[:, 0, 0]
        angle_batch = out[:, 0, 1]

        # out = x
        # for i in range(len(self.nn_model)):
        #     out = self.nn_model[f"layer{i+1}"](out)
        #     if i+1 != len(self.nn_model):  # don't do activation on last layer
        #         print("activation applied")
        #         out = self.relu(out)

        # I think MSEloss() applies an activation function to the last layer itself
        return ClCd_batch, angle_batch


model = ConvNet(input_size, hidden_layers, output_size).to(device)

# loss and optimiser
criterion_ClCd = nn.MSELoss()
criterion_angle = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
model.train()  # needed?
for epoch in tqdm(range(num_epochs)):
    for i, sample in enumerate(train_loader):
        # reshape input to column vector
        sample["coordinates"] = sample["coordinates"].to(device)
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

torch.save(model.state_dict(), print_dir / "model.pkl")  # creates pickle file


def flatten_check(out, targ):
    """check that `out` and `targ` have the same number of elements and flatten them"""
    out, targ = out.contiguous().view(-1), targ.contiguous().view(-1)
    assert len(out) == len(targ), \
        f"Expected output and target to have the same number of elements but got {len(out)} and {len(targ)}."
    return out, targ


def root_mean_square(pred, targ):
    pred, targ = flatten_check(pred, targ)
    return torch.sqrt(F.mse_loss(pred, targ))


def R2_score(pred, targ):
    """R squared score"""
    pred, targ = flatten_check(pred, targ)
    u = torch.sum((targ - pred) ** 2)
    d = torch.sum((targ - targ.mean()) ** 2)
    return 1 - (u / d).item()


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
    ClCd = torch.tensor([])
    angle = torch.tensor([])
    predicted_ClCd = torch.tensor([])
    predicted_angle = torch.tensor([])
    for sample_batched in test_loader:
        sample_batched["coordinates"] = sample_batched["coordinates"].to(device)
        ClCd_batch = sample_batched["y"][:, 0].to(device)
        angle_batch = sample_batched["y"][:, 1].to(device)

        pred_ClCd, pred_angle = model(sample_batched["coordinates"].float())
        ClCd = torch.cat((ClCd_batch, ClCd), 0)
        angle = torch.cat((angle_batch, angle), 0)
        predicted_ClCd = torch.cat((pred_ClCd, predicted_ClCd), 0)
        predicted_angle = torch.cat((pred_angle, predicted_angle), 0)

print(f"ClCd RMS: {root_mean_square(predicted_ClCd, ClCd):.2f}")
print(f"angle RMS: {root_mean_square(predicted_angle, angle):.2f}")

with open(print_dir / "RESULTS.txt", 'w') as f:
    f.write(f"Number of epochs = {num_epochs}\n"
            f"ClCd: RMS = {root_mean_square(predicted_ClCd, ClCd):.2f}, "
            f"R2 = {R2_score(predicted_ClCd, ClCd):.2f}\n"
            f"ClCd: RMS = {root_mean_square(predicted_angle, angle):.2f}, "
            f"R2 = {R2_score(predicted_angle, angle):.2f}\n")

# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv2.register_forward_hook(get_activation('conv2'))
output = model((test_dataset[0])["coordinates"].view(1, 2, 121).float().to(device))

act = activation['conv2'].squeeze()
fig, axarr = plt.subplots(act.size(0))
for idx in range(act.size(0)):
    axarr[idx].plot(act[idx])
# plt.scatter(sample_batched["coordinates"][0, 0, :].detach().numpy(),
#                  sample_batched["coordinates"][0, 1, :].detach().numpy())
plt.show()




