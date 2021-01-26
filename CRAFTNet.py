import sys, os
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, ReLU, Module, Sequential
from torch.autograd import Variable
from torch.optim import adam


class DownConvBlock(Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConvBlock, self).__init__()
        self.conv_layers = Sequential(
            Conv2d(in_channels, out_channels, 3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )
        if pooling:
            self.pooling_layer = MaxPool2d(2, 2)
        else:
            self.pooling_layer = lambda x: x
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pooling_layer(x)
        return x

class UpConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.conv_layers = Sequential(
            Conv2d(in_channels, out_channels*2, 1),
            BatchNorm2d(out_channels*2),
            ReLU(inplace=True),
            Conv2d(out_channels*2, out_channels, 3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        return x
    
class CRAFTNet(torch.nn.Module):
    def __init__(self):
        super(CRAFTNet, self).__init__()
    
        self.downconv1 = DownConvBlock(3, 64)
        self.downconv2 = DownConvBlock(64, 128)
        self.downconv3 = DownConvBlock(128, 256)
        self.downconv4 = DownConvBlock(256, 512)
        self.downconv5 = DownConvBlock(512, 512)
        self.downconv6 = DownConvBlock(512, 512, pooling=False)
        
        self.upconv1 = UpConvBlock(
            self.downconv6._modules['conv_layers'][0].out_channels+self.downconv5._modules['conv_layers'][0].out_channels, 
            256
        )
        self.upconv2 = UpConvBlock(self.downconv4._modules['conv_layers'][0].out_channels+256, 128)
        self.upconv3 = UpConvBlock(self.downconv3._modules['conv_layers'][0].out_channels+128, 64)
        self.upconv4 = UpConvBlock(self.downconv2._modules['conv_layers'][0].out_channels+64, 32)
        
        self.finalConv = Sequential(
            Conv2d(32, 32, kernel_size=3, padding=1), ReLU(inplace=True),
            Conv2d(32, 32, kernel_size=3, padding=1), ReLU(inplace=True),
            Conv2d(32, 16, kernel_size=3, padding=1), ReLU(inplace=True),
            Conv2d(16, 16, kernel_size=1), ReLU(inplace=True),
            Conv2d(16, 2, kernel_size=1),
        )
        
    def forward(self, x):
        #VGG16-BN
        vgg16 = []
        z = self.downconv1(x)
        vgg16.append(z)
        z = self.downconv2(z)
        vgg16.append(z)
        z = self.downconv3(z)
        vgg16.append(z)
        z = self.downconv4(z)
        vgg16.append(z)
        z = self.downconv5(z)
        vgg16.append(z)
        z = self.downconv6(z)
        vgg16.append(z)
        
        #UpConv Network
        z = self.upconv1(torch.cat((z, vgg16[-2]), dim=1))
        z = F.interpolate(z, size=vgg16[-3].size()[2:], mode='bilinear', align_corners=False)
        
        z = self.upconv2(torch.cat((z, vgg16[-3]), dim=1))
        z = F.interpolate(z, size=vgg16[-4].size()[2:], mode='bilinear', align_corners=False)
        
        z = self.upconv3(torch.cat((z, vgg16[-4]), dim=1))
        z = F.interpolate(z, size=vgg16[-5].size()[2:], mode='bilinear', align_corners=False)
        
        z = self.upconv4(torch.cat((z, vgg16[-5]), dim=1))
        feature = z
        
        #Final Conv to Output
        z = F.interpolate(z, size=vgg16[-6].size()[2:], mode='bilinear', align_corners=False)
        y = self.finalConv(z)
        
        return y, feature

