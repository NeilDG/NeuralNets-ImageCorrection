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
from torchvision import models
from matplotlib import pyplot as plt
    
class WarpingCNN(nn.Module):
    
    def __init__(self):
        super(WarpingCNN, self).__init__()
        resnet_model = models.resnet50(True)
        for param in resnet_model.parameters():
            param.requires_grad = False
        
        #self.resnet_bridge = nn.Sequential(*list(resnet_model.children())[:-2])
        
        conv = nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size=3, stride=2, padding=1); nn.init.xavier_normal_(conv.weight)
        pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        relu = nn.ReLU()
        
        self.conv1 = nn.Sequential(conv, pool, relu)
        
        conv = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=2, padding=1); nn.init.xavier_normal_(conv.weight)
        dropout = nn.Dropout2d(p = 0.4)

        self.conv2 = nn.Sequential(conv, pool, relu)
        self.conv3 = nn.Sequential(conv, pool, relu)
        self.conv4 = nn.Sequential(conv, pool, relu)
        self.conv5 = nn.Sequential(conv, pool, relu)
        
        conv = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=2, padding=1); nn.init.xavier_normal_(conv.weight)
        pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        self.conv6 = nn.Sequential(conv, pool, relu)
        self.conv7 = nn.Sequential(conv, pool, relu)
        self.conv8 = nn.Sequential(conv, pool, relu)
        
        conv = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=2, stride=1, padding=1); nn.init.xavier_normal_(conv.weight)
        pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        self.conv9 = nn.Sequential(conv, pool ,relu)
        
        self.fc_block = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                            nn.Linear(32, 16),
                            nn.ReLU(),
                            nn.Linear(16, 2))
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
       
        x = torch.flatten(x,1)
        x = self.fc_block(x)
        
        return x
    
    def print_grad(self, module, grad_input, grad_output):
        print('Inside ' + module.__class__.__name__ + ' backward')
        #print('grad_input size:', grad_input[0].size())
        #print('grad_output size:', grad_output[0].size())
        print('grad_input norm:', grad_input[0].norm())
    
    def visualize_activation(self, module, input, output):
        feature_map = output.cpu().data.numpy()
        a,filter_range,x,y = np.shape(feature_map)
        fig = plt.figure(figsize=(y * 0.07, x * 2))
        #fig = plt.figure()
        
        for i in range(filter_range):
            ax = fig.add_subplot(filter_range, 3, i+1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            
            activation = feature_map[0,i]
            if(i == 0):
                print("Activation shape :", np.shape(activation))
            
            # = cv2.resize(activation, (y * resize_scale, x * resize_scale), interpolation = cv2.INTER_CUBIC)  # scale image up
            ax.imshow(np.squeeze(activation), cmap='gray')
            ax.set_title('%s' % str(i+1))
        
        plt.subplots_adjust(wspace=0, hspace=0.35)
        plt.show() 
        
    def concat_forward(self, x):
        #create 6 different CNNs   
        x1 = self.conv1_block1(x)
        x1 = self.conv2_block1(x1)
        x1 = self.conv3_block1(x1)
        x1 = self.conv4_block1(x1)
        x1 = self.conv_middle1_block1(x1)
        x1 = self.conv5_block1(x1)
        x1 = self.conv_middle2_block1(x1)
        x1 = x1.view(x1.size()[0], -1) #flatten layer
        x1 = self.fc1_block1(x1)
        x1 = self.fc2_block1(x1)
        
        x2 = self.conv1_block2(x)
        x2 = self.conv2_block2(x2)
        x2 = self.conv3_block2(x2)
        x2 = self.conv4_block2(x2)
        x2 = self.conv_middle1_block2(x2)
        x2 = self.conv5_block2(x2)
        x2 = self.conv_middle2_block2(x2)
        x2 = x2.view(x2.size()[0], -1) #flatten layer
        x2 = self.fc1_block2(x2)
        x2 = self.fc2_block2(x2)
        
        x3 = self.conv1_block3(x)
        x3 = self.conv2_block3(x3)
        x3 = self.conv3_block3(x3)
        x3 = self.conv4_block3(x3)
        x3 = self.conv_middle1_block3(x3)
        x3 = self.conv5_block3(x3)
        x3 = self.conv_middle2_block3(x3)
        x3 = x3.view(x3.size()[0], -1) #flatten layer
        x3 = self.fc1_block3(x3)
        x3 = self.fc2_block3(x3)
        
        x4 = self.conv1_block4(x)
        x4 = self.conv2_block4(x4)
        x4 = self.conv3_block4(x4)
        x4 = self.conv4_block4(x4)
        x4 = self.conv_middle1_block4(x4)
        x4 = self.conv5_block4(x4)
        x4 = self.conv_middle2_block4(x4)
        x4 = x4.view(x4.size()[0], -1) #flatten layer
        x4 = self.fc1_block4(x4)
        x4 = self.fc2_block4(x4)
        
        x5 = self.conv1_block5(x)
        x5 = self.conv2_block5(x5)
        x5 = self.conv3_block5(x5)
        x5 = self.conv4_block5(x5)
        x5 = self.conv_middle1_block5(x5)
        x5 = self.conv5_block5(x5)
        x5 = self.conv_middle2_block5(x5)
        x5 = x5.view(x5.size()[0], -1) #flatten layer
        x5 = self.fc1_block5(x5)
        x5 = self.fc2_block5(x5)
        
        x6 = self.conv1_block6(x)
        x6 = self.conv2_block6(x6)
        x6 = self.conv3_block6(x6)
        x6 = self.conv4_block6(x6)
        x6 = self.conv_middle1_block6(x6)
        x6 = self.conv5_block6(x6)
        x6 = self.conv_middle2_block6(x6)
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
    