# Glimpse of the Dataset:

<a href='https://drive.google.com/folderview?id=1RbJHVxo91jhekv3_E9GZvujUDNDaxFQu'>Link to the Dataset</a>
## Steps in Data Creation:

* Collection of the images consisting of:
  * Background Images: 100 images of size 224x224 and image format JPG. The images are mostly of interior of houses.
    Some of them are:
    
    <img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/images/background.png' alt='Background Images'/>
  * Foreground Images: 100 images of random sizes but less than 120 and image format PNG. These images consisted of human beings.
    Some of them are:
    
    <img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/images/Foreground.png' alt='Foreground Images'/>
 
* Background of the foreground images were removed using Power Point
* Masks of the foreground images were generated using gimp.
  Some of them are:
  
  <img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/images/Mask.png' />
  
* Overlay each of the foreground images randomly 20 times on each background images.
* Horizontal Flip the foreground images and again overlap them randomly 20 times on each background images.
* For each overlayed foreground image overlay it's corresponding mask at the same position on a 224x224 black image. This hence generates the mask of the corresponding overlayed images. <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/data_creation-reduced_image_quality.ipynb'>Link to the code</a>
* Now the dataset has a total of:
  * 100 * (100 * 2) * 20 = 400K images of overlayed Foreground-Background images.
    Some of them are: 
    
    <img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/images/fg-bg.png' alt='Fg-Bg Images'/>
 
  * 400K masks of the Foreground-Background images
    Some of them are:
    <img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/images/fg-bg-mask.png' alt='Depth Images'/>
* These 400K images Foreground-Background images were used to produce the 400K depth images.
  Some of them are:
  <img src='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/images/depth.png' alt='Images'/>
* Generate labels file for the dataset in the format:<a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/Labels_Genearation.ipynb'>Link</a>

 | Background_Image_Path | Foreground_Image_Path | Fg-Bg_Image_Path | Fg-Bg-Mask_Image_Path  | Depth_Image_Path |
 | --- | --- | --- | --- | --- |
 | /root_folder/background_img_name.jpg | /root_folder/foregound_img_name.png | /root_folder/fg_bg_img_name.jpg | /root_folder/mask_name.jpg | /root_folder/depth_img_name.jpg |
 
* Compute the mean and std of the Foreground-Background images, their corresponding masks and depth images.<a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/Dataset_Mean_Std.ipynb'>Link</a>

Results:

| Data |	Mean |	STD |
| --- | --- | --- |
| Fg-Bg |	[0.65830478, 0.61511271, 0.5740604 ] |	[0.24408717, 0.2542491, 0.26870159] |
| Fg-Bg-Mask |	[0.04608837, 0.04608837, 0.04608837] |	[0.20544916, 0.20544916, 0.20544916] |
| Depth-Images |	[0.50911522, 0.50911522, 0.50911522] |[0.28174302, 0.28174302, 0.28174302] |

## Final Dataset Folder Structure:

        ├── Foreground
        |   └───── fg1.png
        |   └───── fg2.png
        |   └───── .... 
        |   └───── fg100.png
        |
        |── Background
        |   └───── bg1.jpg
        |   └───── bg2. jpg
        |   └───── ....
        |   └───── bg100.jpg
        |
        |── Mask
        |   └───── mask1.jpg
        |   └───── mask2. jpg
        |   └───── ....
        |   └───── mask100.jpg
        |
        |── Dataset
        |   └───── data_part1.zip
        |   |      └───── data_1
        |   |       |      └───── Fg-Bg
        |   |       |      |     └───── fg-bg <1-80k>.jpg
        |   |       |      └──── Fg-Bg-Mask
        |   |       |      |     └───── fg-bg-mask<1-80k>.jpg
        |   |       |      └──── Depth
        |   |       |      |     └────── depth<1-80k>.jpg\
        |   └───── data_part2.zip
        |   |      └───── data_2
        |   |       |      └───── Fg-Bg
        |   |       |      |     └───── fg-bg <80k-160k>.jpg
        |   |       |      └──── Fg-Bg-Mask
        |   |       |      |     └───── fg-bg-mask<80k-160k>.jpg
        |   |       |      └──── Depth
        |   |       |      |     └────── depth<80k-160k>.jpg
        |   └────── .....
        |
        |   └───── data_part5.zip
        |   |      └───── data_5
        |   |       |      └───── Fg-Bg
        |   |       |      |     └───── fg-bg <320k-400k>.jpg
        |   |       |      └──── Fg-Bg-Mask
        |   |       |      |     └───── fg-bg-mask<320k-400k>.jpg
        |   |       |      └──── Depth
        |   |       |      |     └────── depth<320-400k>.jpg
        |
        └────── labels.txt
 
## Dataset Size:
Total - 3.98GB <br/>
Background Images - 1.2MB <br/>
Foreground Images - 1.2MB <br/>
Mask - 333KB <br/>
Dataset - 3.92GB <br/>
Labels - 52MB


## Some stuffs that helped decrease dataset size:
* Saved all the overlayed images in JPG format.
* Decreased image quality of the overlayed images to 30 from 100.This decreased the dataset size to nearly half.
* Saved all the overlayed masks as a final channel images since the masks were only black n white images.

## <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/others/Overlay_image_on_another.ipynb'>Technique used for Overlaying the images:</a>
* Method 1: Since the background of the Foreground images was transparent, tried to create an image by replacing only those pixels in the background images with the pixels of the foreground images !< 5. This threshold was chosen after considering the fact that the transparent pixel in foreground image was white. This method worked for almost all images but failed for images where there were white patches on the foreground image.
* Method 2: This time i tried to find the boundary of the foreground object but this too failed to create a proper boundary around objects in which there was a part of the foreground object's background enclosed by the object.
* Final Method: In this method the PIL image library function paste was used to overlay the foreground image on the background. This method used the alpha component of the foreground images to overlay them on the background. From this i came to know that the alpha part of a PNG image stores the transparency information of the image.

## <a href='https://github.com/mshilpaa/EVA4/blob/master/Session%2014/Depth_Images_Generation.ipynb'>Generation of Depth Images:</a>
* Code Repository: <a href='https://github.com/ialhashim/DenseDepth'>High Quality Monocular Depth Estimation via Transfer Learning</a>
* Using the trained model from the above repository, we generated the depth images.
* This depth model used an encoder decoder architecture along with transfer learning using DenseNet.
* This depth model used was pretrained on NYU and KITTI dataset.
* For our dataset we used the model weights for the NYU dataset since my dataset was similar to it in the sense that both consisted of images of interior of houses and humans.
* The model accepted images of size 480x640 and the output image size was half the input i.e 240x320. 
* Our images were of size 224x224 and when sent directly to the model gave bad results.
* Resizing the images to 480x640 gave good results.
* But this had the overhead of resizing the input as well as the ouput since the desired output image size was 224x224.
*To avoid atleast one resize tried resizing the images to 448x448 to get the output image size of 224x224. This gave good depth images too.

## Team Members:
1. Shilpa M <br/>
2. Sushmitha M Katti <br/>
3. Deeksha <br/>
4. Noopur <br/>
5. Srinivasan <br/>
We worked as a team and each of us generated 80K each of fg_bg, fg_bg_mask and depth_images
