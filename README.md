[LR_finder]: LR_finder.png

## Aerofoils learner
Convolutional neural network to learn the maximum lift-to-drag ratio at angle for 2D aerofoils. 

Aerofoils downloaded from [Airfoil Tools](airfoiltools.com), with aerodynamic data provided by 
[XFoil](https://web.mit.edu/drela/Public/web/xfoil/).

Unfortunately XFoil has now been depreciated for the latest versions of MacOS, so I am currently unable to get 
aerodynamic data for augmented aerofoils. Note that [Airfoil Tools](airfoiltools.com) provide aerodynamic data from 
[XFoil](https://web.mit.edu/drela/Public/web/xfoil/) on their website (no installation required). This is fundamental as
the current CNN is trained on only 1550 examples.

### To Do
- write the decoder (transpose of convolutions)
- [ShowAerofoil.py](ShowAerofoil.py) needs to be updated for TensorBoard

### Notes
- neural network depreciated. Current work is being completed on the convolutional neural network only
- instructions on how to run XFoil locally to get aerodynamic data found [here](http://airfoiltools.com/airfoil/details?r=polar/index/#xfoil)
- the [aerofoil augmentation](aerofoil_augmentor.py) code can augment new aerofoils. It is suggested that the user plays 
around with the number of control points for thickness and camber and their size with `print_plots = True`. Once the 
number of control points and their lower bound has been decided, turn `print_plots = False` to create new 
aerofoils. Not all the augmented aerofoils will be valid, and a certain of amount of playing around with the parameters 
is required.

## Capabilities
- 2D aerofoil coordinate data with maximum lift-to-drag ratio at angle downloaded with the 
[aerofoils downloader script](download_aerofoils.py)
- aerofoil coordinates redistributed along the x axis (according to a sample x axis distribution) to ensure all inputs 
to the CNN have the same size using the [aerofoil redistribution script](aerofoil_redistribution.py). Moreover, this allows for a reduction in the number of channels (from two to one) as all 
aerofoils have the same x distribution. Therefore, the CNN only takes input from the y coordinates of the aerofoil.
- augmented aerofoils created using the [aerofoil augmentor script](aerofoil_augmentor.py). This uses 
[Bezier curves](https://en.wikipedia.org/wiki/B%C3%A9zier_curve#Explicit_definition) to 
modify the thickness and camber of pre-existing aerofoils using random values. The output files are in the same format 
as the files from the [aerofoil redistribution script](aerofoil_redistribution.py)
- learning rate finder plot available in [run_CNN script](run_CNN.py) by toggling `find_LR = True`
- Convolutional neural network found in [run_CNN](run_CNN.py). Allow learning by toggling `find_LR = False`