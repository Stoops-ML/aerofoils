## Notes
1. aerofoil_redistribution.py not only makes the aerofoils the same size (thereby making the inputs the same size), but 
also allows us to reduce the number of channels from two to one: the x coordinates of all the aerofoils are the same, 
and therefore do not provide any information to the learner (redundant).
2. Shifted training loss to be calculated after 100th epoch and not during the 0th to 100th epoch.
3. Reducing batch size from 100 to 25 helped: Choosing a good minibatch size can influence the learning process
indirectly, since a larger mini-batch will tend to have a smaller variance (law-of-large-numbers) than a smaller
mini-batch. You want the mini-batch to be large enough to be informative about the direction of the gradient, but small
enough that SGD can regularize your network.