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
        
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=6, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, stride = 2, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 48, kernel_size = 2, stride = 2, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4_dropout = nn.Dropout2d(p = 0.5)
        
        self.conv5 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 2, stride = 2, padding = 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv6 = nn.Conv2d(in_channels = 48, out_channels = 48, kernel_size = 2, stride = 2, padding = 1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv7 = nn.Conv2d(in_channels = 48, out_channels = 32, kernel_size = 2, stride = 2, padding = 1)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc = nn.Linear(32, 1)
        
        #init weights
#        nn.init.xavier_uniform_(self.conv1.weight)
#        nn.init.xavier_uniform_(self.conv2.weight)
#        nn.init.xavier_uniform_(self.conv3.weight)
#        nn.init.xavier_uniform_(self.conv4.weight)
#        nn.init.xavier_uniform_(self.conv5.weight)
#        nn.init.xavier_uniform_(self.conv6.weight)
#        nn.init.xavier_uniform_(self.conv7.weight)
#        nn.init.xavier_uniform_(self.fc.weight)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def outputSize(self, in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        return(output)
    
    def forward(self, x):
        #print("Forward pass")
        
        x = F.leaky_relu(self.conv1(x))
        x = self.pool1(x)

        x = F.leaky_relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.leaky_relu(self.conv3(x))
        self.x_store = x
        x = self.pool3(x)
        
        x = F.leaky_relu(self.conv4(x))
        x = self.pool4(x)
        
        x = self.conv4_dropout(x)
        
        x = F.leaky_relu(self.conv5(x))
        x = self.pool5(x)
        self.x_pool_store = x
        
        x = F.leaky_relu(self.conv6(x))
        x = self.pool6(x)
        
        x = F.leaky_relu(self.conv7(x))
        
        x = self.pool7(x)
        
        
        x = x.view(x.size()[0], -1) #flatten layer
        
        x = self.fc(x)
        
        return x
    
    def get_last_layer_activation(self):
        return self.x_store.cpu(), self.x_pool_store.cpu()