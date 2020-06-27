import os
import re
import torch
from pathlib import Path
import numpy as np
from NeuralNets import *
import AerofoilDataset as AD
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# inputs
path = Path(__file__).parent
aerofoils_dir = path / 'data' / 'out' / 'test'
aerofoils = [file for file in os.listdir(aerofoils_dir) if re.search(r"(.csv)$", file)]
model_file = path / 'model.pkl'


class GetActivations:
    def __init__(self):

        # device configuration
        torch.manual_seed(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # paths
        train_dir = path / ('storage/aerofoils' if torch.cuda.is_available() else 'data') / 'out' / 'train'

        # define model parameters
        hidden_layers = [50]
        self.convolutions = [64, 46, 46, 30]  # input/output channels of convolutions
        coords = np.loadtxt(aerofoils_dir / aerofoils[0], delimiter=" ", dtype=np.float32, skiprows=1)  # coords of sample
        input_size = len(coords)
        with open(aerofoils_dir / aerofoils[0]) as f:
            line = f.readline()  # max ClCd & angle are on first line of file
            y_vals = [num for num in re.findall(r'[+-]?\d*[.]?\d*', line) if num != '']  # outputs of sample
            output_size = len(y_vals)
        num_channels = 1  # one channel for y coordinate (xy coordinates requires two channels)

        # create model
        self.model = DenseNet(input_size, output_size, self.convolutions, hidden_layers, num_channels, 32)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()

        # data
        test_dataset = AD.AerofoilDataset(aerofoils_dir, num_channels, input_size, output_size,
                                          transform=transforms.Compose([AD.NormaliseYValues(train_dir)]))
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                 num_workers=4)  # bs=1 required for top_losses()

    def __call__(self, *args, **kwargs):

        def get_activation(name):
            def hook(_, __, output):
                activation[name] = output.detach()

            return hook

        # initialise
        activations = np.zeros((len(self.test_loader), self.convolutions[-1]))

        # get principal components from aerofoils
        for i, (x, _, aerofoil) in enumerate(self.test_loader):
            # hook to get activations
            activation = {}
            self.model.block_and_layer[-1].register_forward_hook(get_activation('final_convolution'))
            output = self.model(x.float().to(self.device))
            act = activation['final_convolution'].squeeze()

            activations[i] = act[:, 0]

        # fit the PCA and transform the original data to it
        principal_components = PCA(n_components=2)
        principal_components.fit(activations)  # fit to 2 eigenvalues (minimise projection error)
        activations_transformed = principal_components.transform(activations)  # reconstruct from compressed representation
        expalined_variance = sum(principal_components.explained_variance_ratio_)

        return activations_transformed, expalined_variance


class PointBrowser(object):
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """

    def __init__(self, principal_components, ax1, ax2):

        # subplots
        self.ax1 = ax1
        self.ax2 = ax2
        self.text = ax.text(0.05, 0.95, 'aerofoil: none', transform=ax.transAxes, va='top')
        self.selected, = self.ax1.plot([principal_components[0, 0]], [principal_components[0, 1]],
                                       'o', ms=12, alpha=0.4, color='yellow', visible=False)

        # aerofoils
        self.chosen_aerofoil_ind = 0
        self.principal_components = principal_components

    def onpress(self, event):
        """use 'n' and 'p' keys to cycle through data"""
        if self.chosen_aerofoil_ind is None:
            return

        if event.key not in ('n', 'p'):
            return

        inc = 1 if event.key == 'n' else -1

        self.chosen_aerofoil_ind += inc
        self.chosen_aerofoil_ind = np.clip(self.chosen_aerofoil_ind, 0, len(self.principal_components[:, 0]) - 1)
        self.update()

    def onpick(self, event):
        """use mouse click to cycle through data"""
        if event.artist != line:
            return True

        N = len(event.ind)
        if not N:
            return True

        # click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        # find closest point
        distances = np.hypot(x - self.principal_components[event.ind, 0], y - self.principal_components[event.ind, 1])
        indmin = distances.argmin()
        self.chosen_aerofoil_ind = event.ind[indmin]
        self.update()

    def update(self):
        """draw selected aerofoil"""
        if self.chosen_aerofoil_ind is None:
            return

        # chosen aerofoil
        chosen_aerofoil = aerofoils_dir / aerofoils[self.chosen_aerofoil_ind]
        coords = np.loadtxt(chosen_aerofoil, delimiter=" ", dtype=np.float32, skiprows=1)  # coords of chosen aerofoil

        # draw aerofoil
        self.ax2.cla()
        self.ax2.title.set_text(f'aerofoil: {aerofoils[self.chosen_aerofoil_ind]}')
        self.ax2.plot(coords[:, 0], coords[:, 1])
        self.ax2.set_ylim([-0.25, 0.25])
        self.selected.set_visible(True)
        self.selected.set_data(self.principal_components[self.chosen_aerofoil_ind, 0],
                               self.principal_components[self.chosen_aerofoil_ind, 1])
        self.text.set_text(f'aerofoil: {aerofoils[self.chosen_aerofoil_ind]}')
        fig.canvas.draw()


if __name__ == '__main__':

    # get principal components of aerofoils
    get_pc = GetActivations()
    pc, explained_var = get_pc()

    # draw figure
    fig, (ax, ax2) = plt.subplots(2, 1)
    fig.suptitle("use mouse or 'n'/'p' keys")
    ax.title.set_text(f"PCA on activations. Explained variance {explained_var:.2f}")
    ax2.title.set_text(f'aerofoil shape')
    line, = ax.plot(pc[:, 0], pc[:, 1], 'o', picker=5)  # 5 points tolerance
    for (x, y), aerofoil in zip(pc, aerofoils):
        ax.text(x, y, aerofoil[:-4])

    # make figure interactive
    browser = PointBrowser(pc, ax, ax2)
    fig.canvas.mpl_connect('pick_event', browser.onpick)
    fig.canvas.mpl_connect('key_press_event', browser.onpress)

    plt.show()
