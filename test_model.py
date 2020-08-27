import torch
import ErrorMetrics as metrics
import matplotlib.pyplot as plt


def test(model, dataloader, dataset, criterion, num_epochs, print_dir, output_size, device='cpu'):
    """test model"""
    model.eval()  # turn off batch normalisation and dropout
    with torch.no_grad():  # don't add gradients of test set to computational graph
        losses = {}
        running_test_loss = 0.
        test_targets_list = torch.tensor([]).to(device)
        test_predictions_list = torch.tensor([]).to(device)

        for test_input, test_targets, aerofoils in dataloader:
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

        running_test_loss /= len(dataset) * 1  # average train loss (=train loss/sample)
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


def activations(model, num_epochs, dataloader, num_activations=None, layer=-1, writer=False, device='cpu'):
    """
    print activations of data
    dataloader is automatically chosen as the test set as it has a batchsize of 1
    activations that are zero (or close to zero) essentially do no work, therefore they're not useful
    """
    #     assert not torch.cuda.is_available():  # not available on cuda

    def get_activation(name):
        def hook(_, __, output):
            activation[name] = output.detach()
        return hook

    # initialise hook
    activation = {}
    activation_name = f'convolutions'
    model.block_and_layer[layer].register_forward_hook(get_activation(activation_name))  # todo fix this

    # get activations
    x, y, aerofoil = next(iter(dataloader))  # using test_loader as it has a batchsize of 1
    _ = model(x.float().to(device))
    act = activation[activation_name].squeeze()

    # plot activations
    if not num_activations:
        num_activations = act.size(0)
    num_activations_plotted = min(act.size(0), num_activations)
    fig, axarr = plt.subplots(num_activations_plotted)
    for idx in range(num_activations_plotted):
        axarr[idx].plot(act[idx])

    if writer:
        writer.add_figure(f"Activations of {activation_name}\nAerofoil {aerofoil[0]}", fig, global_step=num_epochs)
        writer.close()  # todo TB doesn't print image
    else:
        fig.suptitle(f"Activations of {activation_name}\nAerofoil {aerofoil[0]}")
        plt.show()


def heat_map(model, input_size, num_epochs, dataloader=None, sample=None, layer=-1, writer=False, device='cpu'):
    """
    print heat map of a layer (default is last layer)
    default dataloader is test_loader as it has a batchsize of 1
    'sample' includes x, y, aerofoil; as expected from a dataloader
    """
    #     assert not torch.cuda.is_available():  # not available on cuda

    assert dataloader or sample, 'User must provide either a dataloader or a sample'

    def get_activation(name):
        def hook(_, __, output):
            activation[name] = output.detach()
        return hook

    # initialise hook
    activation = {}
    activation_name = 'final_layer'
    model.block_and_layer[layer][-1].register_forward_hook(get_activation(activation_name))  # [-1] is last layer

    # get activations
    if dataloader:
        x, y, aerofoil = next(iter(dataloader))
    else:  # sample provided
        x, y, aerofoil = sample
    _ = model(x.float().to(device))
    act = activation[activation_name].squeeze()

    # heat map of last layer
    average_act = 0
    for channel in act:
        average_act += channel
    average_act /= act.size(0)

    # plot heat map
    fig = plt.figure()
    plt.plot(x.squeeze())
    _ = plt.imshow(average_act.view(1, -1), alpha=0.5, extent=(0, input_size, torch.min(x) * 1.5, torch.max(x) * 1.5),
                   interpolation='bilinear', cmap='magma', aspect='auto')

    if writer:
        writer.add_figure(f"Heat map of last_layer\nAerofoil {aerofoil[0]}", fig, global_step=num_epochs)
        writer.close()  # todo figure not printing to TB (but prints to plt.show)
    else:
        fig.suptitle(f"Heat map of last_layer\nAerofoil {aerofoil[0]}")
        plt.show()
