The goal is to train a model on MNIST dataset to:
* Achieve an accuracy of 99.4 consistently in a no. of epochs.
* Within 15 epochs
* Less than 10k parameters

<a href="https://github.com/mshilpaa/EVA4/blob/master/Session%205/Code%201.ipynb"><h1>Code 1</h1></a>

**Target:**
* Basic model
* Apply batch normalization

**Result:**
* Parameters: 10,272
* Best Train Accuracy: 99.80 (14th Epoch)
* Best Test Accuracy: 99.28 (7th Epoch)

**Analysis:**

* Model is overfitting.

<a href="https://github.com/mshilpaa/EVA4/blob/master/Session%205/Code%202.ipynb"><h1>Code 2</h1></a>

**Target:**
* Since the previous model was overfitting, added dropout of 0.05

**Result:**
* Parameters: 10,272
* Best Train Accuracy: 99.40 (13th Epoch)
* Best Test Accuracy: 99.28 (7th Epoch)

**Analysis:**

* Model is still overfitting but not as much as previous model.
* The gap between train and test accuracy has reduced. Regularization has worked.


<a href="https://github.com/mshilpaa/EVA4/blob/master/Session%205/Code%203.ipynb"><h1>Code 3</h1></a>

**Target:**
* Removed the big kernel in the last layer and used GAP.

**Result:**
* Parameters: 6,836
* Best Train Accuracy: 99.14 (14th Epoch)
* Best Test Accuracy: 99.34 (12th Epoch)

**Analysis:**

* Model is underfitting.
* Train and test accuracy has reduced because the model capacity has reduced from 10,272 parameters to 6,836.
* Capacity of the model can be increased.

<a href="https://github.com/mshilpaa/EVA4/blob/master/Session%205/Code%204.ipynb"><h1>Code 4</h1></a>

**Target:**
* Added another block of conv2d --> relu --> BatchNorm2d --> dropout before GAP.

**Result:**
* Parameters: 8,808
* Best Train Accuracy: 99.10 (13th Epoch)
* Best Test Accuracy: 99.48 (14th Epoch)

**Analysis:**

* Model achieved the target accuracy only once.
* Adding more capacity to the model by introducing an extra layer has shown improvements in test accuracy. Maybe make slight changes to the architecture can push the model to achieve 99.4 consistently in more no. of epochs. Should try increasing the capacity of model a little more.


<a href="https://github.com/mshilpaa/EVA4/blob/master/Session%205/Code%205.ipynb"><h1>Code 5</h1></a>

**Target:**
* Increased capacity of model by using 1x1 after GAP.
* Earlier i was using 3x3 kernel to reduce no. of channels to 10 in the layer before GAP, but now used a 1x1 kernel to do the same. 
* 1x1 acts like a fully connected layer when used after GAP which helps in better classification of the dataset than when used GAP as last layer. So used the 1x1 kernel after GAP. 

**Result:**
* Parameters: 9,844
* Best Train Accuracy: 99.33 (14th Epoch)
* Best Test Accuracy: 99.46 (14th Epoch)

**Analysis:**

* Model is underfitting.
* Using 1x1 after GAP has helped
* Achieved desired test accuracy.But the accuracy is not consistent. The test accuracy is quite irregular after 8th epoch. So should try StepLR. 

<a href="https://github.com/mshilpaa/EVA4/blob/master/Session%205/Code%206.ipynb"><h1>Code 6</h1></a>

**Target:**
*  Used StepLR to change learning rate from 8th epoch onwards 

**Result:**
* Parameters: 9,844
* Best Train Accuracy: 99.41 (13th Epoch)
* Best Test Accuracy: 99.53 (14th Epoch)

**Analysis:**

* Model is underfitting.
* Achieved desired test accuracy consistently.

<a href="https://github.com/mshilpaa/EVA4/blob/master/Session%205/Code%207.ipynb"><h1>Code 7</h1></a>

**Target:**
* Tried to adjust the model to reduce no. of parameters.
* Reduced no. of output channels in the last few layers since mnist is a simple dataset with not many features to learn.
* Used MultiStepLR

**Result:**
* Parameters: 7,332
* Best Train Accuracy: 99.28 (11th Epoch)
* Best Test Accuracy: 99.46 (12th Epoch)

**Analysis:**

* After reducing channel size, observed spots where test accuracy becomes stagnant, so introduced MultiStepLR for learning rate to change from those places.
* The model is underfitting.
* Achieved the desired accuracy within 10k parameters.



