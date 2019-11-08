# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:22:58 2019

@author: delgallegon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_vars as gv
import numpy as np

class ConcatCNN(nn.Module):
    
    def __init__(self):
        super(ConcatCNN, self).__init__()
        
        conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=5, stride=2, padding=1)
        pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        relu = nn.ReLU()
        
        self.conv1_block1 = nn.Sequential(conv1,pool1,relu)
        self.conv1_block2 = nn.Sequential(conv1,pool1,relu)
        self.conv1_block3 = nn.Sequential(conv1,pool1,relu)
        self.conv1_block4 = nn.Sequential(conv1,pool1,relu)
        self.conv1_block5 = nn.Sequential(conv1,pool1,relu)
        self.conv1_block6 = nn.Sequential(conv1,pool1,relu)
        
        conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=2, padding=1)
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.conv2_block1 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block2 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block3 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block4 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block5 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block6 = nn.Sequential(conv2,pool2,relu)
        
        conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=2, padding=1)
        pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.conv3_block1 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block2 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block3 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block4 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block5 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block6 = nn.Sequential(conv3,pool3,relu)
        
        conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.conv4_block1 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block2 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block3 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block4 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block5 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block6 = nn.Sequential(conv4,pool4,relu)
        
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(320, 128)
        self.fc3 = nn.Linear(320, 128)
        self.fc4 = nn.Linear(320, 128)
        self.fc5 = nn.Linear(320, 128)
        self.fc6 = nn.Linear(320, 128)
        
        concat1 = nn.Linear(768, 6)
        tanh = nn.Tanh()
        
        self.concat_block = nn.Sequential(concat1, tanh)
        
    def forward(self, x):
        #create 6 different CNNs   
        x1 = self.conv1_block1(x)
        x1 = self.conv2_block1(x1)
        x1 = self.conv3_block1(x1)
        x1 = self.conv4_block1(x1)
        x1 = x1.view(x1.size()[0], -1) #flatten layer
        x1 = self.fc1(x1)
        
        x2 = self.conv1_block2(x)
        x2 = self.conv2_block2(x2)
        x2 = self.conv3_block2(x2)
        x2 = self.conv4_block2(x2)
        x2 = x2.view(x2.size()[0], -1) #flatten layer
        x2 = self.fc2(x2)
        
        x3 = self.conv1_block3(x)
        x3 = self.conv2_block3(x3)
        x3 = self.conv3_block3(x3)
        x3 = self.conv4_block3(x3)
        x3 = x3.view(x3.size()[0], -1) #flatten layer
        x3 = self.fc3(x3)
        
        x4 = self.conv1_block4(x)
        x4 = self.conv2_block4(x4)
        x4 = self.conv3_block4(x4)
        x4 = self.conv4_block4(x4)
        x4 = x4.view(x4.size()[0], -1) #flatten layer
        x4 = self.fc4(x4)
        
        x5 = self.conv1_block5(x)
        x5 = self.conv2_block5(x5)
        x5 = self.conv3_block5(x5)
        x5 = self.conv4_block5(x5)
        x5 = x5.view(x5.size()[0], -1) #flatten layer
        x5 = self.fc5(x5)
        
        x6 = self.conv1_block6(x)
        x6 = self.conv2_block6(x6)
        x6 = self.conv3_block6(x6)
        x6 = self.conv4_block6(x6)
        x6 = x6.view(x6.size()[0], -1) #flatten layer
        x6 = self.fc4(x6)
        
        y = torch.cat((x1,x2,x3,x4,x5,x6),1)
        
        output = self.concat_block(y)
        
        return output
    