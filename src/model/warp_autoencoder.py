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

class WarpAutoEncoder(nn.Module):
    
    def __init__(self):
        super(WarpAutoEncoder, self).__init__()
        
        """
        Encoder segment
        """
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=4, stride=2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.encoder = nn.Sequential(self.conv1, 
                                     nn.ReLU(self.conv1),
                                     self.conv2,
                                     nn.ReLU(self.conv2))
        
        """
        Decoder segment
        """
        self.conv2_T = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv1_T = nn.ConvTranspose2d(in_channels = 32, out_channels = 3, kernel_size = 4, stride = 2, padding = 1)
        self.decoder = nn.Sequential(self.conv2_T,
                                     nn.ReLU(self.conv2_T),
                                     self.conv1_T,
                                     nn.ReLU(self.conv1_T))
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv1_T.weight)
        nn.init.xavier_uniform_(self.conv2_T.weight)
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        