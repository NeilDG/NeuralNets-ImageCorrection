# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:39:09 2019

Base CNN model using pytorch
@author: delgallegon
"""

import torch.nn as nn
import torch.nn.functional as F
import global_vars as gv

class WarpCNN(nn.Module):
    
    def __init__(self):
        super(WarpCNN, self).__init__()
        
        #Input channels = 3, output channels = 64
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 6, stride = 1, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 8, stride = 1, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        
        self.fc = nn.Linear(405888, 8)
    
    def outputSize(self, in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        return(output)
    
    def forward(self, x):
        #print("Forward pass")
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        #x = x.flatten() #flatten layer
        x = x.view(x.size()[0], -1) #flatten layer
        
        x = self.fc(x)
        
        return x
