The goal is to train a model on MNIST dataset to:
* Achieve an accuracy of 99.4 consistently in a no. of epochs.
* Within 15 epochs
* Less than 10k parameters

**Code 1**

Target:

* Basic model with less than 10k parameters
* Apply batch normalization

Result:

* Parameters: 8,360
* Best Train Accuracy: 99.55 (13th Epoch)
* Best Test Accuracy: 99.19 (13th Epoch)

Analysis:

* Model is overfitting.
* Regularization should be added.

**Code 2**

Target:

Added dropout.
Change no. of output channels of some conv2d layers.
Result:

Parameters: 8,888
Best Train Accuracy: 98.90 (13th Epoch)
Best Test Accuracy: 99.19 (13th Epoch)
Analysis:

The model is underfitting.
The accuracy becomes stagnent after around 8th epoch


**Code 3**

Target:

Used StepLR to change learning rate at 8th epoch

Result:

Parameters: 8,888
Best Train Accuracy: 99.47
Best Test Accuracy: 99.44 (14th Epoch-last epoch)

Analysis:

The model is underfitting.
The target accuracy of 99.4 achieved only once.

**Code 4**

Target:

Added another block of conv2d --> relu --> BatchNorm2d --> dropout
Removed GAP and the last 1x1 layer.
Replaced the 1x1 kernel in last conv2d layer with 2x2 kernel.
Tried to find an architecture similar to the current one with no. of parameters nearest to 10k to achieve accuracy of 99.4 more no. of times.
Result:

Parameters: 12,720
Best Train Accuracy: 99.53 (14th Epoch-last epoch)
Best Test Accuracy: 99.46 (11th Epoch)
Analysis:

The model is not underfitting nor overfitting.
Achieved the desired accuracy but with more parameters.


**Code 5**

Target:

Changed no. of output channels to decrease no. of parameters.Added GAP and 1x1 conv2d layer in the end for the same reason
Result:

Parameters: 9,844
Best Train Accuracy: 99.42 (11th Epoch)
Best Test Accuracy: 99.57 (14th Epoch-last epoch)

**Code 6**

Target:

Changed no. of output channels to decrease no. of parameters.
Used MultiStepLR

Result:

Parameters: 7,332
Best Train Accuracy: 99.28 (11th Epoch)
Best Test Accuracy: 99.46 (12th Epoch)
Analysis:

The model is underfitting.
Achieved the desired accuracy within 10k parameters.

Analysis:

The model is underfitting.
After running the model a few times, realised that accuracy becomes stagnant after certain epochs.So used MultiStepLR to change learning rate at those epochs.
Achieved the desired accuracy within 10k parameters.
