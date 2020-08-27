from NeuralNets import *
import torch


def load_model(checkpoint_path, save_device='cpu', load_device='cpu'):
    """
    load checkpoint from previously run model
    if issues with saving checkpoints from GPU and loading with CPU or GPU check:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    checkpoint = torch.load(checkpoint_path, map_location=save_device)

    # load parameters
    architecture = checkpoint['architecture']
    hyperparameters = checkpoint['hyperparameters']
    extras = checkpoint['extras']

    # create model
    model = DenseNet(architecture['input_size'], architecture['output_size'], architecture['convolutions'],
                     architecture['hidden_layers'], architecture['num_channels'],
                     architecture['dense_out_size']).to(load_device)
    model.load_state_dict(architecture['state_dict'])
    model.to(load_device)

    return model, architecture, hyperparameters, extras


def train(model, optimiser, criterion, num_epochs, train_dataset, valid_dataset, train_loader, valid_loader,
          checkpoint=None, scheduler=False, writer=False, print_every=5, checkpoint_every=None, device='cpu'):
    """train model"""
    checkpoints = []
    if not checkpoint_every and checkpoint:
        checkpoint_every = num_epochs  # checkpoint only last epoch
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

            if (epoch + 1) % print_every == 0 or epoch == 0:
                # training loss
                running_train_loss += train_loss.item() * train_input.shape[0]  # loss.item() returns average loss per sample in batch

                if (i + 1) % len(train_loader) == 0:  # after all batches of training set run

                    # checkpoint
                    if (epoch + 1) % checkpoint_every == 0 and checkpoint:
                        checkpoint['architecture']['state_dict'] = model.state_dict()
                        checkpoint['extras']['epochs'] = epoch + 1
                        checkpoints.append(checkpoint)

                    # validations set
                    with torch.no_grad():  # don't add gradients to computational graph
                        model.eval()
                        for valid_input, valid_targets, _ in valid_loader:
                            # data
                            valid_input = valid_input.to(device)  # y coordinates of aerofoil
                            valid_targets = valid_targets.to(device)  # max ClCd at angle

                            # forward pass
                            valid_predictions = model(valid_input.float())
                            running_valid_loss += criterion(valid_predictions, valid_targets).item() * \
                                                  valid_input.shape[0]

                    # calculate (shifted) train & validation losses (after 1 epoch)
                    running_train_loss /= len(train_dataset) * 1  # average train loss (=train loss/sample)
                    running_valid_loss /= len(valid_dataset) * 1

                    # print to TensorBoard
                    if writer:
                        writer.add_scalar("training loss", running_train_loss,
                                          epoch)  # , epoch * len(train_dataset) + i
                        writer.add_scalar("validation loss", running_valid_loss,
                                          epoch)  # , epoch * len(train_dataset) + i

                    print(f"epoch {epoch + 1}/{num_epochs}, batch {i + 1}/{len(train_loader)}.\n"
                          f"Training loss = {running_train_loss:.4f}, "
                          f"Validation loss = {running_valid_loss:.4f}\n")

                    if scheduler:
                        scheduler.step(running_valid_loss)
                    running_train_loss = 0.
                    running_valid_loss = 0.
                    model.train()

    if writer:
        writer.close()
    # torch.save(model.state_dict(), print_dir / "model.pkl")  # create pickle file

    return model, checkpoints


