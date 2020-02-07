# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 19:01:36 2019

@author: delgallegon
"""

import torch
from torch.utils import data
from loaders import torch_image_dataset as image_dataset
import global_vars as gv
import os
from torchvision import transforms

def assemble_train_data(num_image_to_load = -1):
    rgb_list = []; warp_list = []; warp_list_orig = []; transform_list = []
    
    images = os.listdir(gv.SAVE_PATH_RGB_GT)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len): #len(images)
        rgbImagePath = gv.SAVE_PATH_RGB_GT + images[i]
        rgb_list.append(rgbImagePath)
    
    images = os.listdir(gv.SAVE_PATH_RGB)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
        
    for i in range(image_len):
        imagePath = gv.SAVE_PATH_RGB + images[i]
        warp_list_orig.append(imagePath)
    
    images = os.listdir(gv.SAVE_PATH_WARP)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load * 2
    
    #print("Image len: ", image_len)
    for i in range(image_len):
        if(".png" in images[i]):
            warpImagePath = gv.SAVE_PATH_WARP + images[i]
            transformPath = gv.SAVE_PATH_WARP + images[i].replace(".png", ".txt")
            warp_list.append(warpImagePath)
            transform_list.append(transformPath)
        
    return rgb_list, warp_list_orig, warp_list, transform_list

#if -1, then load all images
def assemble_test_data(num_image_to_load = -1):
    rgb_list = []; warp_list = []; warp_list_orig = []; transform_list = []
    
    images = os.listdir(gv.SAVE_PATH_RGB_GT_VAL)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
    
    for i in range(image_len): #len(images)
        rgbImagePath = gv.SAVE_PATH_RGB_GT_VAL + images[i]
        rgb_list.append(rgbImagePath)
    
    images = os.listdir(gv.SAVE_PATH_RGB_VAL)
    image_len = len(images)
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load
        
    for i in range(image_len):
        imagePath = gv.SAVE_PATH_RGB_VAL + images[i]
        warp_list_orig.append(imagePath)
    
    images = os.listdir(gv.SAVE_PATH_WARP_VAL)
    image_len = int(len(images))
    
    if(num_image_to_load > 0):
        image_len = num_image_to_load * 2
        
    for i in range(image_len):
        if(".png" in images[i]):
            warpImagePath = gv.SAVE_PATH_WARP_VAL + images[i]
            transformPath = gv.SAVE_PATH_WARP_VAL + images[i].replace(".png", ".txt")
            warp_list.append(warpImagePath)
            transform_list.append(transformPath)
        
    return rgb_list, warp_list_orig, warp_list, transform_list

def assemble_unseen_data():
    rgb_list = []; warp_list = []; transform_list = []
    
    images = os.listdir(gv.SAVE_PATH_UNSEEN_DATA_RGB)
    image_len = len(images)
    
    for i in range(image_len): #len(images)
        rgbImagePath = gv.SAVE_PATH_UNSEEN_DATA_RGB + images[i]
        rgb_list.append(rgbImagePath)
    
    images = os.listdir(gv.SAVE_PATH_UNSEEN_DATA_WARP)
    for i in range(image_len * 2):
        if(".png" in images[i]):
            warpImagePath = gv.SAVE_PATH_UNSEEN_DATA_WARP + images[i]
            transformPath = gv.SAVE_PATH_UNSEEN_DATA_WARP + images[i].replace(".png", ".txt")
            warp_list.append(warpImagePath)
            transform_list.append(transformPath)
        
    return rgb_list, warp_list, transform_list

def load_dataset(batch_size = 8, num_image_to_load = -1):
    rgb_list, warp_orig_list, warp_list, transform_list = assemble_train_data(num_image_to_load = num_image_to_load)
    print("Length of train images: ", len(rgb_list), len(warp_list), len(transform_list))
    
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    

    train_dataset = image_dataset.TorchImageDataset(rgb_list, warp_orig_list, warp_list, transform_list, image_transform_op = generic_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )
    
    return train_loader

def load_test_dataset(batch_size = 8, num_image_to_load = -1):
    rgb_list, warp_orig_list, warp_list, transform_list = assemble_test_data(num_image_to_load = num_image_to_load)
    print("Length of test images: ", len(rgb_list), len(warp_orig_list), len(warp_list), len(transform_list))
    
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    test_dataset = image_dataset.TorchImageDataset(rgb_list, warp_orig_list, warp_list, transform_list, image_transform_op = generic_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )
    
    return test_loader

def load_unseen_dataset(batch_size = 8):
    rgb_list, warp_list, transform_list = assemble_unseen_data()
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
        shuffle=False
    )
    
    return test_loader