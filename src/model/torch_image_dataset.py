# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:14:21 2019

Pytorch image dataset loader
@author: delgallegon
"""
import torch
import cv2
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
from ast import literal_eval

class TorchImageDataset(data.Dataset):
    
    def __init__(self, rgb_image_list, warp_image_list, transform_list):
        self.rgb_image_list = rgb_image_list
        self.warp_image_list = warp_image_list
        self.transform_list = transform_list
        
    def __getitem__(self, idx):
        img_id = self.rgb_image_list[idx]
        img = cv2.imread(img_id)
        img = torch.from_numpy(img)
        
        warp_img_id = self.warp_image_list[idx]
        warp_img = cv2.imread(warp_img_id)
        warp_img = torch.from_numpy(warp_img)
        #plt.imshow(warp_img)
        #plt.show()
        
        
        transform_id = self.transform_list[idx]
        contents = np.loadtxt(transform_id)
        #print("Contents:", contents, "Shape: ", np.shape(contents))
        transform_values = torch.tensor(contents)
            
        return img, warp_img, transform_values

    def __len__(self):
        return len(self.rgb_image_list)
        
