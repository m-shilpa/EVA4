
import torch
import numpy as np
from PIL import Image
import os
import torch
# from skimage import io
import numpy as np
from torch.utils.data import Dataset
import random
from matplotlib.image import imread
import matplotlib.pyplot as plt

def unzip_files(File = '/content/gdrive/My Drive/Mask_Rcnn/Dataset'):
  
  from zipfile import ZipFile 
  

  for i in os.listdir(File):
    filename = f'{File}/{i}'
    print(filename)
  # opening the zip file in READ mode 
    with ZipFile(filename, 'r') as zip_file: 
      
        # extracting all the files 
        print('Extracting all the files now...') 
        zip_file.extractall() 
        print('Done!')

def get_data(label_file='/content/gdrive/My Drive/Mask_Rcnn/labels.txt',length=None):
    fg_bg = []
    bg = []
    mask = []
    depth_img = []
    if length == None:
      labels = (open(label_file,'r')).readlines()
    else:
      labels = (open(label_file,'r')).readlines()[:length]
    for label in labels:
      a  = label.split('\t')
      bg.append(f'/content/gdrive/My Drive/Mask_Rcnn/{a[0]}')
      fg_bg.append(f'/content/{a[2]}')         
      mask.append(f'/content/{a[3]}')
      depth = a[4].split('\n')[0]
      depth_img.append(f'/content/{depth}')
    dataset =  list(zip(bg,fg_bg,mask,depth_img))
    random.shuffle(dataset)
    train_split = 70
    train_len = len(dataset)*train_split//100
    train = dataset[:train_len]
    test = dataset[train_len:]
    return train,test

      
class CustomDataset(Dataset):
    def __init__(self, data, transform=None,mask_transform=None,depth_transform=None):
        self.transform = transform
        self.bg,self.fg_bg,self.mask,self.depth_img = zip(*data)
        # self.mask_transform = A.Compose([ A.Normalize(mean=(0.04608837, 0.04608837, 0.04608837)	, 
        #                                                 std=(0.20544916, 0.20544916, 0.20544916)),
        #                                    AP.ToTensor()])
        # self.depth_transform = A.Compose([ A.Normalize(mean=(0.50911522, 0.50911522, 0.50911522)	, 
        #                                                 std=(0.28174302, 0.28174302, 0.28174302)),
        #                                    AP.ToTensor()])

        self.mask_transform = mask_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.fg_bg)

    def __getitem__(self, idx):
  
        # print(self.fg_bg[idx],'---',idx,'----')
        bg = np.asarray(Image.open(self.bg[idx]))
        fg_bg = np.asarray(Image.open(self.fg_bg[idx]))
        mask = np.asarray(Image.open(self.mask[idx]))
        depth = np.asarray(Image.open(self.depth_img[idx]))

        if self.mask_transform:
          img = mask
          # if(len(img.shape) ==2):         
          #       img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

          mask = self.mask_transform(image= img )['image']

        if self.depth_transform:
          img=depth
          # if(len(img.shape) ==2):         
          #       img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
          depth = self.depth_transform(image= img )['image']

        if self.transform:
            bg = self.transform(image=bg)['image']
            fg_bg = self.transform(image=fg_bg)['image']
        
        return bg,fg_bg,mask,depth

def form_data(unzip=False,length =None, train_transform =None,train_mask_transform=None, train_depth_transform=None,
              test_transform =None,test_mask_transform=None, test_depth_transform=None):
  if unzip == True:
    unzip_files()
  else:
    print('Files already downloaded')
  print('Forming the dataset')
  train, test = get_data(length=length)

  train_set = CustomDataset(train,transform=train_transform ,mask_transform=train_mask_transform, depth_transform=train_depth_transform)
  test_set = CustomDataset(test,transform=test_transform ,mask_transform=test_mask_transform, depth_transform=test_depth_transform)
  print('Done!')
  return train_set, test_set