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
        
        self.dropout1_block1 = nn.Dropout2d(p = 0.5)
        self.dropout1_block2 = nn.Dropout2d(p = 0.5)
        self.dropout1_block3 = nn.Dropout2d(p = 0.5)
        self.dropout1_block4 = nn.Dropout2d(p = 0.5)
        self.dropout1_block5 = nn.Dropout2d(p = 0.5)
        self.dropout1_block6 = nn.Dropout2d(p = 0.5)
        
        
        conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=2, padding=1)
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.conv2_block1 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block2 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block3 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block4 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block5 = nn.Sequential(conv2,pool2,relu)
        self.conv2_block6 = nn.Sequential(conv2,pool2,relu)
        
        conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=2, padding=1)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3_block1 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block2 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block3 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block4 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block5 = nn.Sequential(conv3,pool3,relu)
        self.conv3_block6 = nn.Sequential(conv3,pool3,relu)
        
        self.dropout2_block1 = nn.Dropout2d(p = 0.4)
        self.dropout2_block2 = nn.Dropout2d(p = 0.4)
        self.dropout2_block3 = nn.Dropout2d(p = 0.4)
        self.dropout2_block4 = nn.Dropout2d(p = 0.4)
        self.dropout2_block5 = nn.Dropout2d(p = 0.4)
        self.dropout2_block6 = nn.Dropout2d(p = 0.4)
        
        conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        pool4 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        self.conv4_block1 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block2 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block3 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block4 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block5 = nn.Sequential(conv4,pool4,relu)
        self.conv4_block6 = nn.Sequential(conv4,pool4,relu)
        
        conv_middle = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv_middle1_block1 = nn.Sequential(conv_middle, relu)
        self.conv_middle1_block2 = nn.Sequential(conv_middle, relu)
        self.conv_middle1_block3 = nn.Sequential(conv_middle, relu)
        self.conv_middle1_block4 = nn.Sequential(conv_middle, relu)
        self.conv_middle1_block5 = nn.Sequential(conv_middle, relu)
        self.conv_middle1_block6 = nn.Sequential(conv_middle, relu)
        
        conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        pool5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        self.conv5_block1 = nn.Sequential(conv5,pool5,relu)
        self.conv5_block2 = nn.Sequential(conv5,pool5,relu)
        self.conv5_block3 = nn.Sequential(conv5,pool5,relu)
        self.conv5_block4 = nn.Sequential(conv5,pool5,relu)
        self.conv5_block5 = nn.Sequential(conv5,pool5,relu)
        self.conv5_block6 = nn.Sequential(conv5,pool5,relu)
        
        self.dropout3_block1 = nn.Dropout2d(p = 0.25)
        self.dropout3_block2 = nn.Dropout2d(p = 0.25)
        self.dropout3_block3 = nn.Dropout2d(p = 0.25)
        self.dropout3_block4 = nn.Dropout2d(p = 0.25)
        self.dropout3_block5 = nn.Dropout2d(p = 0.25)
        self.dropout3_block6 = nn.Dropout2d(p = 0.25)
        
        conv_middle = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv_middle2_block1 = nn.Sequential(conv_middle, relu)
        self.conv_middle2_block2 = nn.Sequential(conv_middle, relu)
        self.conv_middle2_block3 = nn.Sequential(conv_middle, relu)
        self.conv_middle2_block4 = nn.Sequential(conv_middle, relu)
        self.conv_middle2_block5 = nn.Sequential(conv_middle, relu)
        self.conv_middle2_block6 = nn.Sequential(conv_middle, relu)
        
        self.fc1_block1 = nn.Linear(256, 128)
        self.fc1_block2 = nn.Linear(256, 128)
        self.fc1_block3 = nn.Linear(256, 128)
        self.fc1_block4 = nn.Linear(256, 128)
        self.fc1_block5 = nn.Linear(256, 128)
        self.fc1_block6 = nn.Linear(256, 128)
        
        self.fc2_block1 = nn.Linear(128, 64)
        self.fc2_block2 = nn.Linear(128, 64)
        self.fc2_block3 = nn.Linear(128, 64)
        self.fc2_block4 = nn.Linear(128, 64)
        self.fc2_block5 = nn.Linear(128, 64)
        self.fc2_block6 = nn.Linear(128, 64)
        
        concat1 = nn.Linear(384, 192)
        activation1 = nn.Tanh()
        
        self.concat1_block = nn.Sequential(concat1, activation1)
        
        concat2 = nn.Linear(192, 6)
        
        self.concat2_block = nn.Sequential(concat2)
        
        nn.init.xavier_uniform_(concat1.weight)
        nn.init.xavier_uniform_(self.fc1_block1.weight)
        nn.init.xavier_uniform_(self.fc1_block2.weight)
        nn.init.xavier_uniform_(self.fc1_block3.weight)
        nn.init.xavier_uniform_(self.fc1_block4.weight)
        nn.init.xavier_uniform_(self.fc1_block5.weight)
        nn.init.xavier_uniform_(self.fc1_block6.weight)
        
        nn.init.xavier_uniform_(self.fc2_block1.weight)
        nn.init.xavier_uniform_(self.fc2_block2.weight)
        nn.init.xavier_uniform_(self.fc2_block3.weight)
        nn.init.xavier_uniform_(self.fc2_block4.weight)
        nn.init.xavier_uniform_(self.fc2_block5.weight)
        nn.init.xavier_uniform_(self.fc2_block6.weight)
        
    def forward(self, x):
        #create 6 different CNNs   
        x1 = self.conv1_block1(x)
        x1 = self.dropout1_block1(x1)
        x1 = self.conv2_block1(x1)
        x1 = self.conv3_block1(x1)
        x1 = self.dropout2_block1(x1)
        x1 = self.conv4_block1(x1)
        x1 = self.conv_middle1_block1(x1)
        x1 = self.conv5_block1(x1)
        x1 = self.conv_middle2_block1(x1)
        x1 = self.dropout3_block1(x1)
        x1 = x1.view(x1.size()[0], -1) #flatten layer
        x1 = self.fc1_block1(x1)
        x1 = self.fc2_block1(x1)
        
        x2 = self.conv1_block2(x)
        x2 = self.dropout1_block2(x2)
        x2 = self.conv2_block2(x2)
        x2 = self.conv3_block2(x2)
        x2 = self.dropout2_block2(x2)
        x2 = self.conv4_block2(x2)
        x2 = self.conv_middle1_block2(x2)
        x2 = self.conv5_block2(x2)
        x2 = self.conv_middle2_block2(x2)
        x2 = self.dropout3_block2(x2)
        x2 = x2.view(x2.size()[0], -1) #flatten layer
        x2 = self.fc1_block2(x2)
        x2 = self.fc2_block2(x2)
        
        x3 = self.conv1_block3(x)
        x3 = self.dropout1_block3(x3)
        x3 = self.conv2_block3(x3)
        x3 = self.conv3_block3(x3)
        x3 = self.dropout2_block3(x3)
        x3 = self.conv4_block3(x3)
        x3 = self.conv_middle1_block3(x3)
        x3 = self.conv5_block3(x3)
        x3 = self.conv_middle2_block3(x3)
        x3 = self.dropout3_block3(x3)
        x3 = x3.view(x3.size()[0], -1) #flatten layer
        x3 = self.fc1_block3(x3)
        x3 = self.fc2_block3(x3)
        
        x4 = self.conv1_block4(x)
        x4 = self.dropout1_block4(x4)
        x4 = self.conv2_block4(x4)
        x4 = self.conv3_block4(x4)
        x4 = self.dropout2_block4(x4)
        x4 = self.conv4_block4(x4)
        x4 = self.conv_middle1_block4(x4)
        x4 = self.conv5_block4(x4)
        x4 = self.conv_middle2_block4(x4)
        x4 = self.dropout3_block4(x4)
        x4 = x4.view(x4.size()[0], -1) #flatten layer
        x4 = self.fc1_block4(x4)
        x4 = self.fc2_block4(x4)
        
        x5 = self.conv1_block5(x)
        x5 = self.dropout1_block5(x5)
        x5 = self.conv2_block5(x5)
        x5 = self.conv3_block5(x5)
        x5 = self.dropout2_block5(x5)
        x5 = self.conv4_block5(x5)
        x5 = self.conv_middle1_block5(x5)
        x5 = self.conv5_block5(x5)
        x5 = self.conv_middle2_block5(x5)
        x5 = self.dropout3_block5(x5)
        x5 = x5.view(x5.size()[0], -1) #flatten layer
        x5 = self.fc1_block5(x5)
        x5 = self.fc2_block5(x5)
        
        x6 = self.conv1_block6(x)
        x6 = self.dropout1_block6(x6)
        x6 = self.conv2_block6(x6)
        x6 = self.conv3_block6(x6)
        x6 = self.dropout2_block6(x6)
        x6 = self.conv4_block6(x6)
        x6 = self.conv_middle1_block6(x6)
        x6 = self.conv5_block6(x6)
        x6 = self.conv_middle2_block6(x6)
        x6 = self.dropout3_block6(x6)
        x6 = x6.view(x6.size()[0], -1) #flatten layer
        x6 = self.fc1_block6(x6)
        x6 = self.fc2_block6(x6)
        
#        print("X1: ", np.linalg.norm(x1.cpu().clone().detach()))
#        print("X2: ", np.linalg.norm(x2.cpu().clone().detach()))
#        print("X3: ", np.linalg.norm(x3.cpu().clone().detach()))
#        print("X4: ", np.linalg.norm(x4.cpu().clone().detach()))
#        print("X5: ", np.linalg.norm(x5.cpu().clone().detach()))
#        print("X6: ", np.linalg.norm(x6.cpu().clone().detach()))
#        print()
        y = torch.cat((x1,x2,x3,x4,x5,x6),1)
        
        y = self.concat1_block(y)
        output = y = self.concat2_block(y)
        
        return output
    