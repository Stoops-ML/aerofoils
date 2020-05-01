# Aerofoils Learner
Convolutional neural network to learn the maximum lift-to-drag ratio at angle for 2D aerofoils.

Maximum lift-to-drag ratio at angle is an important aerodynamic property of aircraft wings:
![alt text](https://cdn.comsol.com/wordpress/2015/06/Angle-of-attack-with-lift-and-drag.png "Sample aerofoil")

An aerofoil is a 2D cross-section of a wing. Thus a 3D wing is essentially a combiation of 2D aerofoil sections:
![alt text](https://d2t1xqejof9utc.cloudfront.net/screenshots/pics/76ce7538aace713573297840a447c835/large.PNG "Sample wing")


## Project Goal
Computational fluid dynamics (CFD) is used to find aerodynamic properties of aerofoils such as lift, drag and moment.
There are different levels of fidelity in CFD, where high levels of fidelity requires many more hours, days or weeks 
of computational time. 

The end goal of this project is to take high fidelity aerodynamic data and create a neural network that can accurately 
predict the aerodynamics within a significantly smaller time frame.

Currently low fidelity aerodynamic data is being used in the learner, and in time higher fidelity aerodynamic will be acquired. 
Further, as the project develops 3D wing geometry will replace the 2D aerofoils.  

## Capabilities
- 2D aerofoil coordinate data with maximum lift-to-drag ratio at angle downloaded with the 
[aerofoils downloader script](download_aerofoils.py). This outputs csv files in the correct format for the neural network.
- aerofoil coordinates redistributed along the x axis (according to a sample x axis distribution) to ensure all inputs 
to the neural network have the same size using the [aerofoil redistribution script](aerofoil_redistribution.py). 
Moreover, this allows for a reduction in the number of channels (from two to one) as all 
aerofoils have the same x distribution. Therefore, the CNN only takes the y coordinates of the aerofoil as input.
- augmented aerofoils created using the [aerofoil augmentor script](aerofoil_augmentor.py). This uses 
[Bezier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Explicit_definition) to 
modify the thickness and camber of pre-existing aerofoils using random values. 
This outputs csv files in the correct format for the neural network, but excludes the aerodynamic data. Therefore, these
augmented aerofoils will need to be run through XFoil.
- learning rate finder plot available in [run_CNN script](run_CNN.py) by toggling `find_LR = True`
- convolutional neural network found in [run_CNN script](run_CNN.py). Allow learning by toggling `find_LR = False`

### Resources
- aerofoils downloaded from [Airfoil Tools](airfoiltools.com), with aerodynamic data provided by 
[XFoil](https://web.mit.edu/drela/Public/web/xfoil/).
- instructions on how to run XFoil locally to get aerodynamic data found [here](http://airfoiltools.com/airfoil/details?r=polar/index/#xfoil)
- unfortunately XFoil has now been depreciated for the latest versions of MacOS, so I am currently unable to get 
aerodynamic data for augmented aerofoils. This is fundamental as the current CNN is trained on only 1550 examples. 
Note that [Airfoil Tools](airfoiltools.com) provide aerodynamic data from 
[XFoil](https://web.mit.edu/drela/Public/web/xfoil/) on their website.

### To Do
- [ShowAerofoil.py](ShowAerofoil.py) needs to be updated for TensorBoard
- make print activations method in [run_CNN script](run_CNN.py) more robust
- look into aerofoil transformsations. Currently there is on a flip horizontal transformation (not yet implemented)
- apply PCA on the activations from the convolutions. This will get the most important geometrical features of the aerofoil. From this, plot a graph of two (or three) features against each other for each aerofoil. This should cluster similar aerofoils together and might show interesting latent features.

### Notes
- [run_NN.py](run_NN.py) depreciated. Current work is being completed on the convolutional neural network only
- the [aerofoil augmentation](aerofoil_augmentor.py) code can augment new aerofoils. It is suggested that the user plays 
around with the number of control points for thickness and camber and their size with `print_plots = True`. Once the 
number of control points and their lower bound has been decided, turn `print_plots = False` to create new 
aerofoils. Not all the augmented aerofoils will be valid, and a certain of amount of playing around with the parameters 
is required
