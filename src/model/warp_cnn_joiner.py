# -*- coding: utf-8 -*-
"""
CNN for having a concatenated layer from individual CNNs
Created on Sun Nov  3 21:26:18 2019

@author: delgallegon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import global_vars as gv
import numpy as np

class JoinerCNN(nn.Module):
    
    def __init__(self):
        super(JoinerCNN, self).__init__()
    
        self.concat1 = nn.Linear(1536, 1)
    #maps must be of array size 6    
    def forward(self, maps):
        x = maps[0]
        for i in range(1, len(maps)):
            x = torch.cat((x, maps[i]),1)
        
        x = F.tanh(self.concat1(x))
        
        return x
        