# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:01:36 2019

@author: delgallegon
"""

import torch
from torch.utils import data
from model import torch_image_dataset as image_dataset
import global_vars as gv
import os
from torchvision import transforms

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
    temp_cap = 500 #only load 500 images for faster training
    for i in range(temp_cap): #len(images)
        rgbImagePath = gv.SAVE_PATH_RGB_VAL + images[i]
        rgb_list.append(rgbImagePath)
    
    images = os.listdir(gv.SAVE_PATH_WARP_VAL)
    for i in range(temp_cap * 2):
        if(".png" in images[i]):
            warpImagePath = gv.SAVE_PATH_WARP_VAL + images[i]
            transformPath = gv.SAVE_PATH_WARP_VAL + images[i].replace(".png", ".txt")
            warp_list.append(warpImagePath)
            transform_list.append(transformPath)
        
    return rgb_list, warp_list, transform_list
    print()

def load_dataset(batch_size = 8):
    rgb_list, warp_list, transform_list = assemble_train_data()
    print("Length of train images: ", len(rgb_list), len(warp_list), len(transform_list))
    
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    

    train_dataset = image_dataset.TorchImageDataset(rgb_list, warp_list, transform_list, image_transform_op = generic_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    
    return train_loader

def load_test_dataset(batch_size = 8):
    rgb_list, warp_list, transform_list = assemble_test_data()
    print("Length of test images: ", len(rgb_list), len(warp_list))
    
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    test_dataset = image_dataset.TorchImageDataset(rgb_list, warp_list, transform_list, image_transform_op = generic_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    
    return test_loader