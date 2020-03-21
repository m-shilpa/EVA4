
import torchvision
from torchvision import transforms
import albumentations as A
import albumentations.pytorch as AP
import random
import numpy as np

def load():
	

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

  import numpy as np
  import albumentations as A

  channel_means = (0.5, 0.5, 0.5)
  channel_stdevs = (0.5, 0.5, 0.5)
  train_transform = AlbumentationTransforms([
                                        #  A.Rotate((-30.0, 30.0)),
                                        A.HorizontalFlip(),
                                        #  A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.5),
                                        #  A.ChannelShuffle(always_apply=False, p=0.5), x
                                        #  A.GaussNoise(var_limit=(10.0, 50.0), mean=0.5*255, always_apply=False, p=0.5),
                                        #  A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=0.5),
                                        #  A.Blur(blur_limit=7, always_apply=False, p=0.5),
                                        #  A.RandomScale(scale_limit=0.1, interpolation=1, always_apply=False, p=0.5),
                                        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=20, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
                                        A.Normalize(mean=channel_means, std=channel_stdevs),
                                        A.Cutout(num_holes=2, max_h_size=8,max_w_size = 8,p=1,fill_value=0), # after normalizing as mean is 0, thus fillvalue=0
                                        AP.ToTensor()
                                        ])





  # Test Phase transformations
  test_transform = AlbumentationTransforms([A.Normalize(mean=channel_means, std=channel_stdevs),AP.ToTensor()])
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
  dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

  trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)
  testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

  classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  return classes, trainloader, testloader