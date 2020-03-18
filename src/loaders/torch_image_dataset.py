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
import global_vars as gv

class TorchImageDataset(data.Dataset):
    
    def __init__(self, rgb_image_list, warp_image_list, transform_list, image_transform_op):
        self.rgb_image_list = rgb_image_list
        self.warp_image_list = warp_image_list
        self.transform_list = transform_list
        self.image_transform_op = image_transform_op
        
    def __getitem__(self, idx):
        img_id = self.rgb_image_list[idx]
        img = cv2.imread(img_id); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        
        warp_img_id = self.warp_image_list[idx]
        warp_img = cv2.imread(warp_img_id); warp_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        warp_img = cv2.resize(warp_img, (gv.WARP_W, gv.WARP_H))
        
        if(self.image_transform_op):
            img = self.image_transform_op(img)
            warp_img = self.image_transform_op(warp_img)
        
        transform_id = self.transform_list[idx]
        contents = np.loadtxt(transform_id)
        transform_values = torch.tensor(contents)
        
        file_name = warp_img_id.split("/")[3]
        return img.to(), warp_img, transform_values, file_name
    
    def __len__(self):
        return len(self.rgb_image_list)
        
