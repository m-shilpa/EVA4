
# !pip install -U git+https://github.com/albu/albumentations

import torchvision
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import torch
import random
import numpy as np


	

class AlbumentationTransforms:
  """
  Helper class to create test and train transforms using Albumentations
  """
  def __init__(self, transforms_list=[]):

    self.transforms = A.Compose(transforms_list)


  def __call__(self, img):
    img = np.array(img)
    # print(img)
    return self.transforms(image=img)['image']

def load(train_augmentation=[],test_augmentation=[],mode='default_test_aug',mean=(0.5, 0.5, 0.5),stdev=(0.5, 0.5, 0.5),gpu_batch_size=128 ):
  import numpy as np
  import albumentations as A

  channel_means = mean # r,g,b channels
  channel_stdevs = stdev

  if mode =='default_train_aug' or mode == 'default_aug':
    train_transform = AlbumentationTransforms([
                                          A.Normalize(mean=channel_means, std=channel_stdevs),
                                          AP.ToTensor()
                                          ])
  else:
    train_augmentation.extend([AP.ToTensor()])
    train_transform = AlbumentationTransforms(train_augmentation)





  # Test Phase transformations
  if mode =='default_test_aug' or mode == 'default_aug':
    test_transform = AlbumentationTransforms([A.Normalize(mean=channel_means, std=channel_stdevs),AP.ToTensor()])
  else:
    test_augmentation.extend([AP.ToTensor()])
    test_transform = AlbumentationTransforms(test_augmentation)
  #Get the Train and Test Set
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
  SEED = 1
  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  # For reproducibility
  torch.manual_seed(SEED)
  if cuda:
		  torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=gpu_batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

  trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
  testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

  classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  return classes, trainloader, testloader