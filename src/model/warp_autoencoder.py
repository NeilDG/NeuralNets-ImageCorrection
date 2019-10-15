# -*- coding: utf-8 -*-
"""
Autoencoder architecture for distortion correction
Created on Thu Oct 10 20:14:16 2019

@author: delgallegon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import global_vars as gv
import numpy as np

class WarpAutoEncoder(nn.Module):
    
    def __init__(self):
        super(WarpAutoEncoder, self).__init__()
        
        """
        Encoder segment
        """
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=3, stride=2, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 2, padding = 0)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 1, padding = 0)
        self.encoder = nn.Sequential(self.conv1,
                                     nn.ReLU(self.conv1),
                                     self.conv2,
                                     nn.ReLU(self.conv2),
                                     self.conv3,
                                     nn.ReLU(self.conv3),
                                     self.conv4,
                                     nn.ReLU(self.conv4),
                                     self.conv5,
                                     nn.ReLU(self.conv5))
        
        """
        Decoder segment
        """
        self.conv5_T = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 1, padding = 0)
        self.conv4_T = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv3_T = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv2_T = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 0)
        self.conv1_T = nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size = 3, stride = 2, padding = 0)
        self.decoder = nn.Sequential(self.conv5_T,
                                     nn.ReLU(self.conv5_T),
                                     self.conv4_T,
                                     nn.ReLU(self.conv4_T),
                                     self.conv3_T,
                                     nn.ReLU(self.conv3_T))
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv1_T.weight)
        nn.init.xavier_uniform_(self.conv2_T.weight)
        
    
    def forward(self, x):
        x_skip = F.relu(self.conv1(x))
        x2_skip = F.relu(self.conv2(x_skip))
        x = self.encoder(x)
        x = self.decoder(x)
        
        x = torch.cat((x, x2_skip),1)
        x = F.relu(self.conv2_T(x)) #skip connection from conv2 to conv2_T
        
        x = F.pad(x, (0,1)) #pad to fix dimension
        x = torch.cat((x,x_skip), 1) #skip connection from conv1 to conv1_T
        x = F.relu(self.conv1_T(x))
        
        return x
        