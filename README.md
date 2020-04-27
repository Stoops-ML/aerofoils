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
- create aerofoil augmentation script
- write the decoder (transpose of convolutions)
- [Show aerofoil script](ShowAerofoil.py) needs to be updated for TensorBoard

### Notes
- neural network depreciated. Current work is being completed on the convolutional neural network only
- instructions on how to run XFoil locally to get aerodynamic data found [here](http://airfoiltools.com/airfoil/details?r=polar/index/#xfoil)

## Methodology
- 2D aerofoil coordinate data with maximum lift-to-drag ratio at angle downloaded with the 
[aerofoils downloader script](download_aerofoils.py)
- aerofoil coordinates redistributed along the x axis (according to a sample x axis distribution) to ensure all inputs 
to the CNN have the same size. Moreover, this allows for a reduction in the number of channels (from two to one) as all 
aerofoils have the same x distribution. Therefore, the CNN only takes input from the y coordinates of the aerofoil.
- learning rate finder plot: ![alt text][LR_finder]