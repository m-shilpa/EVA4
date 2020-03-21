

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



def show_random_images(dataset, classes):

	# get some random training images
	dataiter = iter(dataset)
	images, labels = dataiter.next()

	img_list = range(5, 9)

	# show images
	print('shape:', images.shape)
	imshow(torchvision.utils.make_grid(images[img_list],nrow=len(img_list)))
	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in img_list))


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# functions to show an image
def imshow1(img,c = "airp"):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(7,7))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title(c)

def show_random_images_classwise(dataset, classes):

	# get some random training images

  dataiter = iter(dataset)
  images, labels = dataiter.next()
  for i in range(10):
    index = [j for j in range(len(labels)) if labels[j] == i]
    imshow1(torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=2,scale_each=True),classes[i])

def show_classwise_images(dataset, classes,classname=None,Range=range(5,9)):

	# get some random training images
  dataiter = iter(dataset)
  images, labels = dataiter.next()

 
  img_list = []
  if classname ==None:
    img_list= Range
  else:
    for i in range(len(labels)):
      if classes[labels[i]]==classname:
        img_list.append(i)
  
  print('Shape:', images.shape)
  img = torchvision.utils.make_grid(images[img_list],nrow=2)
  img = img / 2 + 0.5     # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
	# print labels

  print(' '.join('%5s' % classes[labels[j]] for j in img_list))
  print(img_list)