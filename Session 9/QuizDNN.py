import torch
import torch.nn as nn
import torch.nn.functional as F


class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.05),
            
            )

        self.pool = nn.MaxPool2d(2, 2)
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.FC =  nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)     

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1+x2)
        x4 = self.pool(x1+x2+x3) 
        x5 = self.conv4(x4) 
        x6 = self.conv5(x4+x5)
        x7 = self.conv6(x4+x5+x6)
        x8 = self.pool(x5+x6+x7) 
        x9 = self.conv7(x8)
        x10 = self.conv8(x8+x9)
        x11 = self.conv9(x8+x9+x10)
        x12 = self.GAP(x11)
        x13 = self.FC(x12)  
        x = x13.view(-1, 10)
        return x


