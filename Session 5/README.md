The goal is to train a model on MNIST dataset to:
* Achieve an accuracy of 99.4 consistently in a no. of epochs.
* Within 15 epochs
* Less than 10k parameters

# Code 1

**Target:**
* Basic model
* Apply batch normalization

**Result:**
* Parameters: 10,272
* Best Train Accuracy: 99.80 (14th Epoch)
* Best Test Accuracy: 99.28 (7th Epoch)

**Analysis:**

* Model is overfitting.

# Code 2

**Target:**
* Since the previous model was overfitting, added dropout of 0.05

**Result:**
* Parameters: 10,272
* Best Train Accuracy: 99.40 (13th Epoch)
* Best Test Accuracy: 99.28 (7th Epoch)

**Analysis:**

* Model is still overfitting.
* Reached the desired accuracy only once.


# Code 3

**Target:**
* Removed the big kernel in the last layer and used GAP.

**Result:**
* Parameters: 6,836
* Best Train Accuracy: 99.14 (14th Epoch)
* Best Test Accuracy: 99.34 (12th Epoch)

**Analysis:**

* Model is underfitting.
* Capacity of the model can be increased.

# Code 4

**Target:**
* Added another block of conv2d --> relu --> BatchNorm2d --> dropout

**Result:**
* Parameters: 8,808
* Best Train Accuracy: 99.10 (13th Epoch)
* Best Test Accuracy: 99.48 (14th Epoch)

**Analysis:**

* Model achieved the target accuracy only once.

# Code 5

**Target:**
*  Instead of using 3x3 kernel to reduce no. of channels to 10 in the layer before GAP, use a 1x1 kernel. 
* When 1x1 used after GAP acts like an FC

**Result:**
* Parameters: 9,844
* Best Train Accuracy: 99.33 (14th Epoch)
* Best Test Accuracy: 99.46 (14th Epoch)

**Analysis:**

* Model is underfitting.
* Achieved desired test accuracy.But the accuracy is not consistent. The test accuracy is quite irregular after 8th epoch. So should try StepLR. 

# Code 6

**Target:**
*  Used StepLR

**Result:**
* Parameters: 9,844
* Best Train Accuracy: 99.41 (13th Epoch)
* Best Test Accuracy: 99.53 (14th Epoch)

**Analysis:**

* Model is underfitting.
* Achieved desired test accuracy consistently.

# Code 7

**Target:**
* Reduced no. of output channels in the last few layers since mnist is a simple dataset with not many features to learn.
* Used MultiStepLR

**Result:**
* Parameters: 7,332
* Best Train Accuracy: 99.28 (11th Epoch)
* Best Test Accuracy: 99.46 (12th Epoch)

**Analysis:**

* After reducing channel size observed stagnent stop in test accuracy, so introduced MultiStepLR for learning rate to change from those places.
* The model is underfitting.
* Achieved the desired accuracy within 10k parameters.



