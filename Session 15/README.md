# Mask and Depth Prediction

## Problem Statement:
Build a dnn model that predicts the depth map as well as a mask for the foreground object given background and foreground-background images.

<h2><a href='https://github.com/mshilpaa/EVA4/tree/master/Session%2014'> Dataset: </a></h2>

  * 100 Background images
  * 400K Foreground-Background images ( 100 foreground images placed at 20 random locations on the 100 background images )
  * 400K Mask images of the Foreground-Background images
  * 400K Depth images of the Foreground-Background images
  
## Implementation:
  
### Gist:
* <b>Model:</b> Resnet-Unet 
* <b>Input size:</b> 64x64x6 - bg and fgbg concatenated in the z-axis.
* <b>Output sizes:</b> ( 64x64x1, 64x64x1 ) --> ( mask, depth )
* <b>Total params:</b> 11,234,066
* <b>Trainable params:</b> 11,234,066
* <b>Non-trainable params:</b> 0
* <b>Input size (MB):</b> 0.09
* <b>Forward/backward pass size (MB):</b> 121.19
* <b>Params size (MB):</b> 42.85
* <b>Total Size (MB):</b> 164.14
* <b>Time Taken/epoch:</b> 17 minutes
* <b>Epochs:</b> 20
* <b>Loss Function for Mask prediction:</b> BCEWithLogitsLoss
* <b>Loss Function for Depth prediction:</b> SSIM
* <b>Optimizer:</b> SGD with momentum
* <b>Learing Rate:</b> StepLR starting at 0.01

## <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/eva_files/customdata_loading.py'> Building the dataloader:</a>
### Workflow:
* First the data files are unzipped.
* Next, built a method that prepares a list which contains strings that indicate the location of the image. The list is then shuffled randomly and split into train_set and test_set in the ratio 7:3.
* Then built the class CustomDataset() that applied augmentations to the train and test data. The images are opened when the augmentation needs to be applied. It returns the images as a tuple - (bg,fgbg,mask,depth)

## Data Augmentations:
The augmentations I applied to the dataset are: 
* Fg-Bg and Bg – Resize(from 224x224 to 64x64), Cutout,Normalize,ToTensor
* Mask,depth – Resize,ToTensor
* The other augmentations I tried were RandomContrast and ToGray. These did not increase performance of the network and in fact increased the loss.
* I had to resize my images since bigger size images needed more memory and reducing the batch size to make this possible would increase training time. So i resized my images to 64x64.
* I didn’t apply much augmentation to the input images. Augmentations like scaling, cropping could not be applied since we have to do depth estimation of the entire image. Other distortions in the input image might also affect the depth images prediction.  




## Jouney towards building the final DNN Model :

* Initially i started off going through articles related to segmentation, like what models are used, how they are trained, what loss is used, how accuracy is calculated. 
* I had many doubts like how an image can be genrated from the dnn since till now we were only predicting labels, how loss is propagated backward, how to give 2 inputs to the model should it be given separately or concatenated, what model to use...
* While going through many articles i came across siamese network which took two images as input and gave the differnece between the two input images as the output. So this network have a 'Y'-like architecture. Since the input to my model was also fg-bg and bg images i thought fg_bg - bg would give me the fg image. So i wanted to try this architecture. But before this i manually tried doing the fg_bg -bg calculation and the output was not the fg but a mess which had not even a little essence of the original images. So dropped the idea.
* Next i thought of implementing a network like the region proposal network in mask-rcnn. Mask-rcnn produced two outputs, the bounding boxes and the mask. The RPN produced feature-rich output that it could be used to prdict masks too. In a similar way i thought of producing depth images too from the RPN. When i started building the model, there were a lot of confusions like how many layers, number of channels, the RPN in mask-rcnn used bounding boxes too, so i had to build one without it and the deadline was coming near too.
* With a lot of these ideas and confusions i simply tried resnet18 to see what would be the result. I tried it for both depth and mask. To my surprise the masks predicted was fine but the depth images were bad but not as much as what i had imagined. 
* The monocular depth estimation model which was used to generate our ground truth depth images used an encoder-decoder architecture.So i tried an encoder-decoder architecture for the depth. The performance was good.
* At this point, i got a clearer idea of things and lot less confusions.Also the deadline had got extended too.
* For building the model TensorBoard helped me a lot. I could clearly see all the layers and their connections in tensorboard allowing me to check and correct my mistakes.
* Next i combined both the models and trained them in a desire to see pretty good results. But the end result was good depth images but completely nothing for the mask. This i suppose was due to the way loss was backpropagated.
* So i thought of trying the encoder-decoder network even to the mask.
* While browsing through many I came across unet which was mentioned in many of the websites but till then i couldn't try it due to time constraints. But later when the assignment submission date got postponed, i decided to try unet. 
* So finally i built my unet architecture with 11 lakh parameters.


<h2><a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/eva_files/dnn_architecture.py'>Architecture Details:</a> </h2>



## Resnet-Unet : 

* U-net was originally invented and first used for biomedical image segmentation. Its architecture can be thought of as an encoder network followed by a decoder network.
  * The encoder is the first half in the architecture. It usually is a classification network like VGG/ResNet where you apply convolution blocks followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels.
  *	The decoder is the second half of the architecture. Image reconstrution is performed here. The goal is to semantically project the discriminative features (lower resolution) learnt by the encoder onto the pixel space (higher resolution) to get a dense classification. This is what makes unet powerful. The decoder consists of upsampling and concatenation followed by convolution blocks.

* I used Resnet blocks in the Unet since the skip connections could allow the network to learn the small objects too and project them to the output.
* My dnn architecture consists of 1 encoder block and 2 decoder blocks – one for predicting the mask and the other for predicting depth. So each of them will be specialized for predicting their respective outputs. Following is a high-level represntation of my dnn architecture.

<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/arch1.png' alt='simple representation of the model' width=400/>

<details>
 <summary>A Detailed Representation of Model</summary>
 
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/unet-arch-2.png' alt='simple representation of the model'  width=1000/>
 
</details>



* For downsampling, a convolution layer with kernel size 3 and stride 2 is used. Downsampling reduces the size of the input but increases the no. of channels compensating for the decrease in width and height of input
* For upsampling, a transpose convolution layer is used with a kernel size 2. Upsampling increases the width and height of the input and reduces no. of channels, performing the exact opposite task of downsampling. Transpose convolution increases the image (x,y) dimensions also maintaining a connectivity between the input and the output to it.Following demonstrates how the input is upsampled using transpose convolution.
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/transpose_conv.gif' alt='transpose convolution' width=400 />


* Other upsampling techniques that could be used are:
  * Bilinear: Uses all nearby pixels to calculate the pixel's value, using linear interpolations.
  * Bicubic: It also uses all nearby pixels to calculate the pixel's values, through polynomial interpolations. Usually produces a smoother surface than the previous techniques.


<a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/eva_files/training_testing_losses.py'><h2>Training:</h2> </a>
* The input to my model was the fg-bg, bg concatenated in z-axis.
* I trained my model on a sample of the dataset. I used 80k images each (bg,fgbg,mask,depth), based on the result i later trained the model on the entire dataset using the final list of hyperparameters. Doing this took lesser time and allowed me to try different combinations of the hyperparameters.

<h3>Saving the model:</a></h3>
My internet speed was too low and sometimes it took almost 3 times the normal speed of execution for each epoch. So I saved my best model and loaded them later to continue execution. In addition to saving the model I also saved the loss values of the mask and depth to analyse the results. 

<h2> <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/eva_files/training_testing_losses.py'>Loss functions:</a> </h2>
I tried many loss functions but the ones which performed well were: Mask – BCEWithLogitsLoss , Depth – SSIM

### <b>BCEWithLogitsLoss:</b> 
In this loss function, sigmoid is applied on the target followed by BCELoss.
This is Binary cross entropy loss mostly used when the number of classes is 2. Here, i used BCEWithLogitsLoss for the mask since we can think of the background and foreground as the two classes of the mask image.
The following shows the computation of BCEWithLogitsLoss:
<img src= 'https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/bce.png' alt='bce'/>

### <b>SSIM – Structural SIMilarity :</b>
This is a perceptual loss that measures the similarity between two inputs based on the luminance (I), contrast (C), and structure (S) of the two images.
Patches of the images are taken and compared. 
In pixel-wise comparison each pixel of the image1 is compared with corresponding pixels of image2.But in SSIM, a window size is choosen, say 5. So  now 5 pixels from all sides of the single pixel is taken to finally form a 11x11 patch. This patch is now compared with the corresponding patch in the other image. The comparison here is not pixel-wise but luminance (I), contrast (C), and structure (S) – wise. 
The following is the formula for ssim:  
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/ssim-formula-1.png' alt='ssim1'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/ssim-formula-2.png' alt='ssim2'/>

The loss value returned by SSIM is the structural dissimilarity between the inputs: <br/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/ssim-formula-3.png' alt='ssim3'/>
 <br/>

## Plots of the Loss :
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/plot_avg_loss.png' width=300 /><img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/plot_depth.png' width=300 /><img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/plot_mask.png' width=300 />

## Accuracy metric:
<h3>Depth images:</h3>
Pixel-wise comparison or IOU may not be a good metric for measuring the accuracy of the network in predicting depth images.They give a quantitative measure. But usually when we look at the output of the network, we decide on the quality of the output based on it's appearance.Depth images have variations in the brightness and contrast. We can use the perception-based quality metric, SSIM index, to measure the quality of the depth image. 

<h3>Mask images:</h3> 

Dice Coefficient can be the evaluation metric here. 

Dice Coefficient = ( 2 * Area of Overlap ) / total number of pixels in both images.

The mask consists of the background and foreground. The foreground is only 1 object. So Dice Coefficient between the target and the predicted images can be a good metric for evaluating the predicted mask

<h2><a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/Final_Session_15.ipynb'>Results:</a></h2>

* <b>Training:</b>
  * Mask Loss: 0.0365
  * Depth Loss: 0.0329
  * Avg. Loss: 0.0347
* <b>Testing:</b>
  * Mask Loss: 0.0340
  * Depth Loss: 0.0420
  * Avg. Loss: 0.0380
  * Dice Coefficient: 0.7517
  * SSIM Index: 0.9158
* <h3>Output of the network: </h3> 
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/final-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/final-depth.JPG' alt='ssim'/>

<h2><a href='https://github.com/mshilpaa/EVA4/tree/master/Session%2015/others'> Other Loss functions I tried and their result :</a></h2>


<h3>  Mask: MSELoss, Depth: MSELoss </h3>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/mse-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/mse-depth.JPG' alt='ssim'/>
<h3>Mask: SmoothL1 Loss, Depth: SSIM</h3>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/smoothl1_ssim-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/smoothl1_ssim-depth.JPG' alt='ssim'/>
<h3> Mask: SSIM, Depth: SSIM</h3>

<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/ssim-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/ssim-depth.JPG' alt='ssim'/>

There were many more combination of the losses i implemented, images of whose i happened to not store them.
 
Some points on the losses:
* The above images were generated after training the model for 20 epochs each on 80K images each (bg,fgbg,mask,depth).
* MSELoss blurs the output although the final loss value was pretty low. This may be because training using MSELoss follows a trajectory similar to linear interpolation. Alternative to MSELoss was the perceptual losses like SSIM which compares based on the quality similar to the human vision system.
* SmoothL1 loss was also similar to MSELoss. This maybe due to the similarity in the way the two losses are calculated. 
* Losses based on pixel to pixel comparison didn't seem to work well for predicting depth. 
* A combination of the losses didn't produce extraordinary results. So using a single loss function each of the depth and mask seemed enough.
* Using the same loss function for both didn't give good results.
* I used different window sizes for SSIM but these didn't give good results either.
* Finally the one that gave good results was BCEWithLogitsLoss and SSIM.
* After trying so many loss functions i truly felt like a loss engineering like how Rohan had told.

## Code Files: 
* <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/Final_Session_15.ipynb'>Final Notebook</a>
* <a href='https://github.com/mshilpaa/EVA4/tree/master/Session%2015/eva_files'>Code modules</a>
* <a href='https://github.com/mshilpaa/EVA4/tree/master/Session%2015/others'>Implementation of other Losses</a>

## References:
* <a href='https://towardsdatascience.com/u-net-b229b32b4a71'>UNet</a>
* <a href='http://www.cs.toronto.edu/~jsnell/assets/perceptual_similarity_metrics_icip_2017.pdf'>LEARNING TO GENERATE IMAGES WITH PERCEPTUAL SIMILARITY METRICS</a>
* <a href='https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0'>Up-sampling with Transposed Convolution</a>
* <a href='https://discuss.pytorch.org/'>https://discuss.pytorch.org/</a>
* <a href='https://torchgeometry.readthedocs.io/en/latest/losses.html'>Kornia</a>


  
