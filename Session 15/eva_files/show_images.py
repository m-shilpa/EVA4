

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

  
def show_misclassified_images(model, device, test_loader,classes,num=25):
  correct = 0
  misclassify = []
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()
          
          for i in range(len(pred)):
            if pred[i] != target[i]:              
              if len(misclassify)==0:
                misclassify.append([data[i]])
                misclassify.append([classes[pred[i]]])
                misclassify.append([classes[target[i]]])
              else:
                misclassify[0].append(data[i])
                misclassify[1].append(classes[pred[i]])
                misclassify[2].append(classes[target[i]])

  j=0
  fig = plt.figure(figsize=(15,15)) 
  for i in range(num): 
      ax = fig.add_subplot(5,5 , 1 + j) # 4 rows, 3 columns, 1+j is the index which gives position of each image in the plot
      imshow(misclassify[0][i].cpu()) # display the image
      title = "Predicted: "+ str(misclassify[1][i])+" Target: "+ str(misclassify[2][i])
      ax.set_title(title) # give the class of the image as its title
      j+=1
  plt.subplots_adjust( hspace=0.5, wspace=0.35)      
  plt.show()
  return misclassify # in the format[[image],[predict],[target]]

def show_unet_images(actual,predicted,title=''):
  import torchvision
  import torchvision
  img1 = torchvision.utils.make_grid(actual[:8].unsqueeze(1).detach().cpu(),1)
  img1_1 = img1.permute(2,1,0)


  img2 = torchvision.utils.make_grid(predicted[:8].detach().cpu(),1)
  img2_1 = img2.permute(2,1,0)

  fig, axs = plt.subplots(1,2,figsize=(20,20))
  axs[0].imshow(img1_1)
  axs[0].set_title(f'Actual {title}')
  axs[1].imshow(img2_1)
  axs[1].set_title(f'Predicted {title}')