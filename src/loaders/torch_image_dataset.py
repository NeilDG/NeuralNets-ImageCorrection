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
    
    def __init__(self, rgb_image_list, warp_orig_list, warp_image_list, transform_list, image_transform_op):
        self.rgb_image_list = rgb_image_list
        self.warp_orig_list = warp_orig_list
        self.warp_image_list = warp_image_list
        self.transform_list = transform_list
        self.image_transform_op = image_transform_op
        
    def __getitem__(self, idx):
        img_id = self.rgb_image_list[idx]
        img = cv2.imread(img_id); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        #img = torch.from_numpy(img)
        
        warp_orig_id = self.warp_orig_list[idx]
        warp_orig_img = cv2.imread(warp_orig_id); warp_orig_img= cv2.cvtColor(warp_orig_img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        warp_orig_img = cv2.resize(warp_orig_img, (gv.WARP_W, gv.WARP_H))
        
        warp_img_id = self.warp_image_list[idx]
        warp_img = cv2.imread(warp_img_id); warp_img = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB) #because matplot uses RGB, openCV is BGR
        warp_img = cv2.resize(warp_img, (gv.WARP_W, gv.WARP_H))
        #warp_img = torch.from_numpy(warp_img)
        
        if(self.image_transform_op):
            img = self.image_transform_op(img)
            warp_orig_img = self.image_transform_op(warp_orig_img)
            warp_img = self.image_transform_op(warp_img)
        
        
        #img = img.type(torch.FloatTensor)
        #warp_img = img.type(torch.FloatTensor)
        
        transform_id = self.transform_list[idx]
        contents = np.loadtxt(transform_id)
        #print("Contents:", contents, "Shape: ", np.shape(contents))
        transform_values = torch.tensor(contents)
          
        print("Path: ", warp_orig_id, warp_img_id)
        return img.to(), warp_orig_img, warp_img, transform_values

    def __len__(self):
        return len(self.rgb_image_list)
        
