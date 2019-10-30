# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:39:09 2019

Base CNN model using pytorch
@author: delgallegon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import global_vars as gv

class WarpCNN(nn.Module):
    
    def __init__(self):
        super(WarpCNN, self).__init__()
        
        self.layer_activations = []; self.pool_activations = []
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        #self.conv1_dropout = nn.Dropout2d(p = 0.5)
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        #self.conv2_dropout = nn.Dropout2d(p = 0.5)
        
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        #self.conv4_dropout = nn.Dropout2d(p = 0.5)
        
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv6 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv7 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc = nn.Linear(64, 1)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
        self.layer_activations = [0, 1, 2, 3];
        self.pool_activations = [0, 1, 2, 3];
        
        self.flag = False
    
    def outputSize(self, in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
        return(output)
    
    def forward(self, x):
        #print("Forward pass")
        
        x = F.relu(self.conv1(x))
        if(self.flag):
            self.layer_activations[0] = x.cpu().clone().detach()
        
        x = self.pool1(x)
        if(self.flag):
            self.pool_activations[0] = x.cpu().clone().detach()

        #x = self.conv1_dropout(x)
        
        x = F.relu(self.conv2(x))
        if(self.flag):
            self.layer_activations[1] = x.cpu().clone().detach()
        
        x = self.pool2(x)
        if(self.flag):
            self.pool_activations[1] = x.cpu().clone().detach()
        
        #x = self.conv2_dropout(x)
        
        x = F.relu(self.conv3(x))
        if(self.flag):
            self.layer_activations[2] = x.cpu().clone().detach()
        
        x = self.pool3(x)
        if(self.flag):
            self.pool_activations[2] = x.cpu().clone().detach()
        
        x = F.relu(self.conv4(x))
        if(self.flag):
           self.layer_activations[3] = x.cpu().clone().detach()
        
        x = self.pool4(x)
        if(self.flag):
            self.pool_activations[3] = x.cpu().clone().detach()
        
        #x = self.conv4_dropout(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        
        x = F.relu(self.conv7(x))
        x = self.pool7(x)
        
        x = x.view(x.size()[0], -1) #flatten layer
        
        x = F.tanh(self.fc(x))
        
        return x
    
    def flag_visualize_layer(self,flag):
        self.flag = flag
        
    def get_layer_activation(self, index):
        return self.layer_activations[index], self.pool_activations[index]