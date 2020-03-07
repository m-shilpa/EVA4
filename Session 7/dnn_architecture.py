import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.05),
            
            )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),
            
            )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, groups=32), 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(1, 1), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),
            
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False,groups=48), 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(1, 1), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),
            
            )
        self.convblock4 = nn.Sequential(
            
            
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=1, bias=False,dilation =2), 
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(p=0.05),

            

            nn.Conv2d(in_channels=48, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), 

            
            )

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.convblock1(x) # i/p= 32 o/p=32 r=3 
        x = self.pool(x) # i/p = 32 o/p = 16
        x = self.convblock2(x) # 16
        x = self.pool(x) # 8
        x = self.convblock3(x) # 8
        x = self.pool(x) # 4
        x = self.convblock4(x)  # 4
        x = self.gap(x)

        x = x.view(-1, 10)

        return x


# net = Net()