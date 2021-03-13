import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class DepthSepConv2d(nn.Module):
    """ An implementation of the Depth-wise seperable convolutional layers described in the paper 
        "MobileNets: Efficient Convolutional Neural Networks for Mobile VisionApplications" """
    def __init__(self, chan_in, chan_out, kernel_size,stride):
        super().__init__()
        self.dwconv = DepthwiseConv2d(chan_in, kernel_size=kernel_size,stride=stride)
        self.bnorm1  = nn.BatchNorm2d(chan_in)
        self.pwconv = nn.Conv2d(chan_in, chan_out, kernel_size=1, stride=1)
        self.bnorm2  = nn.BatchNorm2d(chan_out)
    
    def forward(self, x):
        x = F.relu(self.bnorm1(self.dwconv(x)))
        x = F.relu(self.bnorm2(self.pwconv(x)))
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, chan_in, kernel_size, stride):
        super().__init__()
        self.kernels = [nn.Conv2d(1,1, kernel_size=kernel_size, padding=kernel_size//2, \
                        padding_mode='reflect',stride=stride).to(device) for _ in range(chan_in)]
        
    def forward(self, x):
        channels = [kernel(x[:,chan:chan+1,:,:]) for chan, kernel in enumerate(self.kernels)]
        return torch.cat(channels,1)

