# -*- coding: utf-8 -*-
"""
Experiment to perform correction of image using pixel-wise error
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
    
class PixelwiseCNN(nn.Module):
    
    def __init__(self):
        super(PixelwiseCNN, self).__init__()
        
        conv = nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size=3, stride=1, padding=1); nn.init.xavier_normal_(conv.weight)
        relu = nn.ReLU()
        
        self.conv1 = nn.Sequential(conv, relu)
        
        conv = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size=3, stride=1, padding=1); nn.init.xavier_normal_(conv.weight)
        
        self.conv2 = nn.Sequential(conv, relu)
        self.conv3 = nn.Sequential(conv, relu)
        self.conv4 = nn.Sequential(conv, relu)
        
        conv = nn.Conv2d(in_channels = 128, out_channels = 3, kernel_size=3, stride=1, padding=1); nn.init.xavier_normal_(conv.weight)
        sigmoid = nn.Sigmoid()
        self.conv5 = nn.Sequential(conv, sigmoid)
        
    
    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + res #perform residual
        x = self.conv4(x)
        x = self.conv5(x)
        #print("Forward shape:" , np.shape(x))
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
    