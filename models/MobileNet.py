import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.DepthSepConv2d import DepthSepConv2d

class MobileNet(nn.Module):
    """ An implementation of the light weigh image recognition architecture described 
        in the 2017 paper "MobileNets: Efficient Convolutional Neural Networks for 
        Mobile Vision Applications" by Howard et al. """
    def __init__(self, n_classes , alpha=1, rho=1):
        super().__init__()
        self.n_classes = n_classes
        # Initial Ordinary Convolutional Layer
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1, padding_mode='reflect')
        # Intermediate 3x3 Depth-Wise Separable Layers
        self.codw1 = DepthSepConv2d(32,64,kernel_size=3,stride=1)
        self.codw2 = DepthSepConv2d(64,128,kernel_size=3,stride=2)
        self.codw3 = DepthSepConv2d(128,128,kernel_size=3,stride=1)
        self.codw4 = DepthSepConv2d(128,256,kernel_size=3,stride=2)
        self.codw5 = DepthSepConv2d(256,256,kernel_size=3,stride=1)
        self.codw6 = DepthSepConv2d(256,512,kernel_size=3,stride=2)
        # Repeated Constant Resolution 3x3 Depth-Wise Separable Layers
        self.codw7 = DepthSepConv2d(512,512,kernel_size=3,stride=1)
        self.codw8 = DepthSepConv2d(512,512,kernel_size=3,stride=1)
        self.codw9 = DepthSepConv2d(512,512,kernel_size=3,stride=1)
        self.codw10 = DepthSepConv2d(512,512,kernel_size=3,stride=1)
        self.codw11 = DepthSepConv2d(512,512,kernel_size=3,stride=1)
        # 2 Stride 2 3x3 Depth-Wise Separable Layers
        self.codw12 = DepthSepConv2d(512,1024,kernel_size=3,stride=2)
        self.codw13 = DepthSepConv2d(1024,1024,kernel_size=3,stride=1)
        # Global Average Pool and Fully connected output layer
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024,n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.codw1(x)
        x = self.codw2(x)
        x = self.codw3(x)
        x = self.codw4(x)
        x = self.codw5(x)
        x = self.codw6(x)
        x = self.codw7(x)
        x = self.codw8(x)
        x = self.codw9(x)
        x = self.codw10(x)
        x = self.codw11(x)
        x = self.codw12(x)
        x = self.codw13(x)
        x = self.avgpool(x)
        x = x.view(-1,1024)
        x = self.fc(x)
        return x
