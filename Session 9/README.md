Target : Train a RESNET18 model to achieve 87% accuracy by applying augmentation strategies. Implement gradcam.

**Results:**

Test Accuracy: 90 

**Augmentation stratagies used :**

Train Data:

* Horizontal Flip
* Cutout
* ToTensor
* Normalize

Test Data:

* ToTensor
* Normalize

**Gradcam:**

Produces a heatmap showing what the kernel looks at to classify the image at a particular layer. Gradient of a particular predicted class w.r.t the channel pixel values is taken for all the channels in that layer. Then mean of these values for each channel is taken and the corresponding channels are multiplied with this value.On doing this the channels that most affect the prediction of the particular class amplifies and vis versa. Then finally the many channels are converting to one channel by adding the depth. Then relu is applied to this channel to remove negative values. The resulting channel values are used to produce a heatmap which is overlaped on the input image. 

<img src="https://github.com/mshilpaa/EVA4/blob/master/Session%209/gradcam.jpg"/>

**Gradcam results for cifar10:**

<img src="https://github.com/mshilpaa/EVA4/blob/master/Session%209/gradcam_cifar10.jpg"/>
