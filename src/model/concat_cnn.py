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
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=5, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        
        #self.conv1_dropout = nn.Dropout2d(p = 0.5)
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        
        #self.conv2_dropout = nn.Dropout2d(p = 0.5)
        
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.fc = nn.Linear(256, 64)
        self.concat1 = nn.Linear(384, 1)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, x):
        #create 6 different CNNs
        intermediate_tensors = []
        
        for i in range(6):
            x1 = self.pool1(F.relu(self.conv1(x)))
            x1 = self.pool2(F.relu(self.conv2(x1)))
            x1 = self.pool3(F.relu(self.conv3(x1)))
            x1 = self.pool4(F.relu(self.conv4(x1)))
            x1 = x1.view(x1.size()[0], -1) #flatten layer
            x1 = self.fc(x1)
            
            intermediate_tensors.append(x1)
        
        y = intermediate_tensors[0]
        for i in range(1, len(intermediate_tensors)):
            y = torch.cat((y, intermediate_tensors[i]),1)
        
        y1 = F.tanh(self.concat1(y))
        y2 = F.tanh(self.concat1(y))
        y3 = F.tanh(self.concat1(y))
        y4 = F.tanh(self.concat1(y))
        y5 = F.tanh(self.concat1(y))
        y6 = F.tanh(self.concat1(y))
        
        outputs = [y1,y2,y3,y4,y5,y6]
        outputs = torch.Tensor(outputs)
        outputs.requires_grad_()
        return 
    