# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:28:09 2019

Image and tensor utilities
@author: delgallegon
"""

import numpy as np

def convert_to_matplotimg(img_tensor, batch_idx):
    img = img_tensor[batch_idx,:,:,:].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0) #for properly displaying image in matplotlib
    
    return img
    
