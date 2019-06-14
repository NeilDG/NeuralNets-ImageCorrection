# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:01:36 2019

@author: delgallegon
"""

import torch
import cv2
from torch.utils import data
import global_vars as gv
import os

def assemble_train_data():
    rgb_list = []; warp_list = []; transform_list = []
    
    images = os.listdir(gv.SAVE_PATH_RGB)
    temp_cap = 500 #only load 500 images for faster training
    for i in range(temp_cap): #len(images)
        rgbImagePath = gv.SAVE_PATH_RGB + images[i]
        rgb_list.append(rgbImagePath)
    
    images = os.listdir(gv.SAVE_PATH_WARP)
    for i in range(temp_cap * 2):
        if(".png" in images[i]):
            warpImagePath = gv.SAVE_PATH_WARP + images[i]
            transformPath = gv.SAVE_PATH_WARP + images[i].replace(".png", ".txt")
            warp_list.append(warpImagePath)
            transform_list.append(transformPath)
        
    return rgb_list, warp_list, transform_list

def assemble_test_data():
    rgb_list = []; warp_list = []; transform_list = []
    
    images = os.listdir(gv.SAVE_PATH_RGB_VAL)
    for i in range(len(images)):
        rgbImagePath = gv.SAVE_PATH_RGB_VAL + images[i]
        rgb_list.append(rgbImagePath)
    
    images = os.listdir(gv.SAVE_PATH_WARP_VAL)
    for i in range(len(images)):
        if(".png" in images[i]):
            warpImagePath = gv.SAVE_PATH_WARP_VAL + images[i]
            transformPath = gv.SAVE_PATH_WARP_VAL + images[i].replace(".png", ".txt")
            warp_list.append(warpImagePath)
            transform_list.append(transformPath)
        
    return rgb_list, warp_list, transform_list
    print()
