import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # i/p=32 o/p=32 r=3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), # i/p=32 o/p=32 r=5 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.05),
            
            )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # i/p=16 o/p=16 r=10
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # i/p=16 o/p=16 r=14 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),
            
            )

        self.convblock3 = nn.Sequential(

            # Depthwise Separable Convolution
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, groups=32), # i/p=8 o/p=8 r=24 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(1, 1), padding=1, bias=False), # i/p=8 o/p=8 r=24 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            # Depthwise Separable Convolution
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False,groups=48), # i/p=8 o/p=8 r=32 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 1), padding=1, bias=False), # i/p=8 o/p=8 r=32 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False), # i/p=8 o/p=8 r=40 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False), # i/p=8 o/p=8 r=48 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),
            
            )
        self.convblock4 = nn.Sequential(
            
            # Dilated Convolution
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False,dilation =2), # i/p=4 o/p=4 r=68  
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            

            nn.Conv2d(in_channels=48, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # i/p=4 o/p=4 r=84 

            
            )

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):

        x = self.convblock1(x) # i/p=32 o/p=32 r(at the end of block)=5 

        x = self.pool(x)       # i/p=32 o/p=16 r=6

        x = self.convblock2(x) # i/p=16 o/p=16 r=14

        x = self.pool(x)       # i/p=16 o/p=8 r=16

        x = self.convblock3(x) # i/p=8 o/p=8 r=48

        x = self.pool(x)       # i/p=8 o/p=4 r=52

        x = self.convblock4(x) # i/p=4 o/p=4 r=84

        x = self.gap(x)        

        x = x.view(-1, 10)

        return x


