

import torch
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



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
       

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# calling Resnet18
def Resnet18():
    return ResNet(BasicBlock, [2,2,2,2])
    

class New_Resnet(nn.Module):
    def __init__(self):
        super(New_Resnet,self).__init__()

        self.prepLayer = nn.Sequential(
                    nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),bias=False,padding=1,stride=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                )
        
        self.cmbr_block1 = self._CMBR_Block(64,128)
        self.res_block1 = self._ResBlock(128,128)

        self.layer2 = self._CMBR_Block(128,256)

        self.cmbr_block2 = self._CMBR_Block(256,512)
        self.res_block2 = self._ResBlock(512,512)

        self.MP4x4 = nn.MaxPool2d(4,4)

        self.fc = nn.Conv2d(in_channels=512,out_channels=10,kernel_size=(1,1),bias=False,padding=0,stride=1)
    
    def _ResBlock(self,in_channels,out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),bias=False,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),bias=False,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        
    def _CMBR_Block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),bias=False,padding=1,stride=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self,x):
      # PrepLayer
      x = self.prepLayer(x)
      # Layer 1
      x = self.cmbr_block1(x)
      r1 = self.res_block1(x)
      x = x + r1
      # Layer 2
      x = self.layer2(x)
      # Layer 3
      x = self.cmbr_block2(x)
      r2 = self.res_block2(x)
      x = x + r2

      x = self.MP4x4(x)
      x = self.fc(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)



class Resnet_TinyImageNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*2*2, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.linear(out)
        return out

# calling Resnet18
def Resnet18_TinyImageNet(num_classes):
    return Resnet_TinyImageNet(BasicBlock, [2,2,2,2],num_classes=num_classes)
    



