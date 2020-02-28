# -*- coding: utf-8 -*-
"""
CNN with RESNET pre-trained
Created on Wed Nov  6 19:22:58 2019

@author: delgallegon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_vars as gv
import numpy as np
from torchvision import models

class ResNetCNN(nn.Module):
    
    def __init__(self):
        super(ResNetCNN, self).__init__()
        
        self.resnet_model = models.resnet50(True)
        for param in self.resnet_model.parameters():
            param.requires_grad = False
        
        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(num_ftrs, 6)
       
    
    def forward(self, x):
        x = self.resnet_model(x)
        return x
    