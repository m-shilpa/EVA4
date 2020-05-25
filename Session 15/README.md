# Mask and Depth Prediction

## Problem Statement:
Build a model that predicts the mask and depth when Background images and Foreground-Background images are given as input.

## Dataset:
  * 100 Background images
  * 400K Foreground-Background images 
  * 400K Mask images 
  * 400K Depth images
  
## Implementation:
  
## <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/eva_files/customdata_loading.py'> Building the dataloader:</a>
### Workflow:
* First the data files are unzipped.
* Prepared a list which contains strings that indicate the location of the image. The list is then shuffled randomly and split into train_set and test_set in the ratio 7:3.
* Then built the class CustomDataset() that applied augmentations to the train and test data. The images are opened when the augmentation needs to be applied. It returns the tuple of images - (bg,fgbg,mask,depth)

## Data Augmentations:
The augmentations I applied to the dataset are: 
* Fg-Bg and Bg – Resize(to 64x64), Cutout,Normalize,ToTensor
* Mask,depth – Resize,ToTensor
* The other augmentations I tried were RandomContrast and ToGray. These did not increase performance of the network and in fact increased the loss.
* I didn’t apply much augmentation to the input images since we have to do depth estimation of the entire image. So distortions in the input image might affect the depth images prediction.  

## Building the DNN Model :

* Initially i started off going through articles related to segmentation, like what models are used, how they are trained, what loss is used, how accuracy is calculated. Here I came across unet which mentioned in many of the websites.
So I decided to try using unet as my architecture for predicting masks. The monocular depth estimation model which was used to generate our ground truth depth images used an encoder-decoder architecture similar to unet. So i decided to try using unet for predicting depth images too.
* Initially I started off implementing a separate architecture for predicting the depth and mask images. I built a single unet architecture but trained individually for the depth and mask. This I tried to check if my model works for the dataset.
* I heavily used tensorBoard while building my model to check for the correctness of the connections.

<h2><a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/eva_files/dnn_architecture.py'>Architecture Details:</a> </h2>

## Unet: 

* U-net was originally invented and first used for biomedical image segmentation. Its architecture can be thought of as an encoder network followed by a decoder network.
  * The encoder is the first half in the architecture. It usually is a classification network like VGG/ResNet where you apply convolution blocks followed by a maxpool downsampling to encode the input image into feature representations at multiple different levels.
  *	The decoder is the second half of the architecture. The goal is to semantically project the discriminative features (lower resolution) learnt by the encoder onto the pixel space (higher resolution) to get a dense classification. The decoder consists of upsampling and concatenation followed by convolution blocks.

* I used a Resnet blocks in the Unet
* So my model architecture consists of 1 encoder block and 2 decoder blocks – one for predicting the mask and other for predicting depth. So each of them will be specialized for predicting their respective outputs.

<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/arch.png' alt='simple representation of the model' width=400/>

<details>
 <summary>A Detailed Representation of Model</summary>
 
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/unet-arch-2.png' alt='simple representation of the model'  width=400/>
 
</details>



* I used resnet blocks since the skip connections could allow the network to learn the small objects too.
* For downsampling, a convolution layer with kernel size 3 and stride 2 is used. Downsampling reduces the size of the input but increases the no. of channels compensating for the decrease in width and height of input
* For upsampling, a transpose convolution layer is used with a kernel size 2. Upsampling increases the width and height of the input and reduces no. of channels performing the exact opposite task of downsampling. Transpose convolution increases the image (x,y) dimensions also maintaining a connectivity between the input and the output to it.Following demonstrates how the input is upsampled using transpose convolution.
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/transpose_conv.gif' alt='transpose convolution' width=400 />


* Other upsampling techniques that could be used are:
  * Bilinear: Uses all nearby pixels to calculate the pixel's value, using linear interpolations.
  * Bicubic: It also uses all nearby pixels to calculate the pixel's values, through polynomial interpolations. Usually produces a smoother surface than the previous techniques.
  
## Model Stats:  

* Total params: 11,234,066
* Trainable params: 11,234,066
* Non-trainable params: 0
* Input size (MB): 0.09
* Forward/backward pass size (MB): 121.19
* Params size (MB): 42.85
* Total Size (MB): 164.14
* Time Taken/epoch: 17 minutes

<h2><a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/eva_files/training_testing_losses.py'>Saving the model:</a></h2>
My internet speed was too low and it took almost 3 times the normal speed of execution for each epoch. So I saved my best model and loaded them later to continue execution. In addition to saving the model I also saved the loss values of the mask and depth to analyse the results. 

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
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/plot_avg_loss.png' width=200 /><img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/plot_depth.png' width=200 /><img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/plot_mask.png' width=200 />

## Loss functions I tried and their result :
###  BCEWithLogitsLoss , SSIM
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/final-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/final-depth.JPG' alt='ssim'/>
<h3>  MSELoss : </h3>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/mse-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/mse-depth.JPG' alt='ssim'/>
<h3>SmoothL1 Loss, SSIM</h3>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/smoothl1_ssim-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/smoothl1_ssim-depth.JPG' alt='ssim'/>
<h3> SSIM:</h3>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/ssim-mask.JPG' alt='ssim'/>
<img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/images/ssim-depth.JPG' alt='ssim'/>

## Accuracy metric:
<h3>Depth images:</h3>
Pixel-wise comparison or IOU may not be a good metric for measuring the accuracy of the network in predicting depth images. Depth images have these variations in the brightness and contrast. We can use the perception-based quality metric, SSIM index, to measure the quality of the depth image. 

<h3>Mask images:</h3> 

Dice Coefficient can be the evaluation metric here. 
Dice Coefficient = ( 2 * Area of Overlap ) / total number of pixels in both images.
The mask consists of the background and foreground. The foreground is only 1 object. So Dice Coefficient between the target and the predicted images can be a good metric for evaluating the predicted mask

## Refernces: 
* <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2015/Final_Session_15.ipynb'>Final Notebook</a>
* <a href='https://github.com/mshilpaa/EVA4/tree/master/Session%2015/eva_files'>Code Files</a>
* <a href='https://github.com/mshilpaa/EVA4/tree/master/Session%2015/others'>Implementation of other Losses</a>



  
