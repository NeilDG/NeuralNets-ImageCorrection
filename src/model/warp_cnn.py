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
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size=8, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 8, stride = 1, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 8, stride = 1, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 6, stride = 1, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 6, stride = 1, padding = 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv6 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 6, stride = 1, padding = 1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv7 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv8 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv9 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 1)
        self.pool9 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dropout_conv3 = nn.Dropout2d(p = 0.5)
        self.dropout_conv6 = nn.Dropout2d(p = 0.5)
        self.dropout_conv9 = nn.Dropout2d(p = 0.5)
        self.fc = nn.Linear(16, 8)
        
        #init weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.xavier_uniform_(self.conv9.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        
        #init biases
#        nn.init.constant_(self.conv1.bias, -1.0)
#        nn.init.constant_(self.conv2.bias, -1.0)
#        nn.init.constant_(self.conv3.bias, -1.0)
#        nn.init.constant_(self.conv4.bias, -1.0)
#        nn.init.constant_(self.conv5.bias, -1.0)
#        nn.init.constant_(self.conv5.bias, -1.0)
#        nn.init.constant_(self.conv6.bias, -1.0)
#        nn.init.constant_(self.conv7.bias, -1.0)
#        nn.init.constant_(self.conv8.bias, -1.0)
#        nn.init.constant_(self.conv9.bias, -1.0)
#        nn.init.constant_(self.fc.bias, 0.4)
    
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
        
        x = self.dropout_conv3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        
        x = self.dropout_conv6(x)
        
        x = F.relu(self.conv7(x))
        x = self.pool7(x)
        
        x = F.relu(self.conv8(x))
        x = self.pool8(x)
        
        x = F.relu(self.conv9(x))
        x = self.pool9(x)
        
        x = self.dropout_conv9(x)
        
        #x = x.flatten() #flatten layer
        x = x.view(x.size()[0], -1) #flatten layer
        
        x = self.fc(x)
        
        return x
