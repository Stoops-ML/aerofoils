## Notes
1. aerofoil_redistribution.py not only makes the aerofoils the same size (thereby making the inputs the same size), but 
also allows us to reduce the number of channels from two to one: the x coordinates of all the aerofoils are the same, 
and therefore do not provide any information to the learner (redundant).
2. Shifted training loss to be calculated after 100th epoch and not during the 0th to 100th epoch.
3. Reducing batch size from 100 to 25 helped: Choosing a good minibatch size can influence the learning process
indirectly, since a larger mini-batch will tend to have a smaller variance (law-of-large-numbers) than a smaller
mini-batch. You want the mini-batch to be large enough to be informative about the direction of the gradient, but small
enough that SGD can regularize your network.


- I don't think I need to do a batchnorm after every convolution because the coordinates are already normalised by chord

======
Increasing the number of convolutions significantly helped the learner. I couldn't find anything else that would help 
the learner improve (LR, number of hidden layers etc.) - only increasing the number of convolutions worked! Note that I 
reduced stride down to 1 (so output size = input size after every convolution), but this didn't affect learning by 
itself. Note that the LR finder shows that I should be using a LR of 1, but this was a very bad choice for this 
configuration: a LR of 0.1 was much better. However, it's underfitting. Increasing batch size prevented this, but this 
changed the bounds of the train and validation losses. Perhaps increase batch size at a later stage. A third 
convolutional layer doesn't give any improvement, in fact it performs similarly.

I don't think the LR finder works at all.

changing optimiser to Adam, it gives a slightly better result than SGD. No real improvement with more than 300 hidden 
units in first layer. Adding a third convolutional layer doesn't reduce training loss much. Increasing number of 
channels in convolutional layers improves loss slightly (but really not much). Using few sample in batch to generalise 
better - it's not very computationally expensive.

I've split up the outputs from the forward method, now neural network works a lot better: improvement of 400%. This 
explaines why the LR finder was so bad. However, now the forward method outputs a tuple and not a tensor the LR finder 
does not work. I've spoken to the developers and they're looking into allowing it to work with a tuple.

LR finder now fixed: I've created my own wrapper for multiple loss functions (see https://github.com/davidtvs/pytorch-lr-finder/issues/35)

test set probably has a lower loss than training and validation losses because dropout is deactivated. This means that
all the weights/activations are used in the model, not just some of them (as determined by the probability in dropout).

y values are now normalised (x values already normalised by chord length) for train and validation sets, but not for 
the test set. This means the test set root mean square value is the true RMS (because the test data isn't normalised).
It's best to use [-1,1] min-max scaling or zero-mean, unit-variance standardization. Scaling your data into [0,1] will 
result in slow learning (I SHOULD THEREFORE PROBABLY NORMALISE X VALUES). You must use the same normalisation scaling for train, valid and test sets.