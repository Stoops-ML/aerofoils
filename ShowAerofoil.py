import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def show_aerofoil(writer, tensorboard=False, **kwargs):
    """show plot of aerofoil"""
    fig_aero = plt.figure()
    plt.plot(kwargs["coordinates"], 'r-')
    ClCd, angle = kwargs["y"]
    plt.title(f"{kwargs['aerofoil']}\n"
              f"Max ClCd = {ClCd:.2f} at {angle:.2f} degrees")
    if tensorboard:
        writer.add_figure(kwargs["aerofoil"], fig_aero)
        writer.close()
    else:
        plt.show()


def show_aerofoil_batch(batch_num, **sample_batched):
    """show plot of aerofoils for a batch of samples."""
    # TODO: tidy up matplotlib plotting of batches
    # TODO: add TensorBoard functionality. Put all plots in a grid using
    # images = np.reshape(train_images[0:25], (-1, 28, 28, 1))  # Don't forget to reshape.
    # tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

    aerofoils_batch, coordinates_batch, y_batch = sample_batched['aerofoil'], sample_batched['coordinates'],\
                                                  sample_batched["y"]
    ClCd_batch, angle_batch = y_batch[:, 0], y_batch[:, 1]
    batch_size = len(aerofoils_batch)

    fig = plt.figure()
    for i, (aerofoil, coords, ClCd, angle) in enumerate(zip(aerofoils_batch, coordinates_batch, ClCd_batch,
                                                            angle_batch)):
        ax = fig.add_subplot(1, batch_size, i+1)
        ax.plot(coords, 'r-')
        plt.title(f"{aerofoil}\n"
                  f"Max ClCd = {ClCd:.2f} at {angle:.2f} degrees")

    plt.suptitle(f'Batch #{batch_num} from dataloader')
    plt.show()
