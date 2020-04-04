# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:28:09 2019

Image and tensor utilities
@author: delgallegon
"""

import numpy as np
import cv2

def convert_to_matplotimg(img_tensor, batch_idx):
    img = img_tensor[batch_idx,:,:,:].numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0) #for properly displaying image in matplotlib
    
    return img

def convert_to_opencv(img_tensor):
    img = img_tensor.numpy()
    img = np.moveaxis(img, -1, 0)
    img = np.moveaxis(img, -1, 0) #for properly displaying image in matplotlib
    
    return img

#loads an image compatible with opencv
def load_image(file_path):
    img = cv2.imread(file_path)
    if(img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    else:
        print("Image ",file_path, " not found.")
    return img
    
