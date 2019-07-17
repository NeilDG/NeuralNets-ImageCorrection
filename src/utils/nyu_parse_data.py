# -*- coding: utf-8 -*-

"""
Code to parse NYU image data to PNG files
Created on Mon Feb 11 20:44:41 2019

@author: NeilDG
"""

import numpy as np
import h5py
from matplotlib import pyplot as plt
import cv2

# data path
path_to_depth = 'C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/nyu_depth_v2_labeled.mat'
SAVE_PATH_RGB = 'C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/nyu_image_rgb/'
SAVE_PATH_DEPTH = 'C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/nyu_image_depth/'
def start_parse():
    print("Start parse of NYU")
    # read mat file
    f = h5py.File(path_to_depth)
    index = 0;
    
    # read 0-th image. original format is [3 x 640 x 480], uint8
    for img in f["images"]: 
        # reshape
        img_ = np.empty([480, 640, 3])
        img_[:,:,0] = img[0,:,:].T
        img_[:,:,1] = img[1,:,:].T
        img_[:,:,2] = img[2,:,:].T
        
        # imshow
        img__ = img_.astype('float32')
        img__ = img__/255.0
        #plt.imshow(img__)
        #plt.show()
        
        cv2.imwrite(SAVE_PATH_RGB + "rgb_" +str(index)+ ".png", img_)
        
        # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
        depth = f['depths'][index]
        
        # reshape for imshow
        depth_ = np.empty([480, 640, 3])
        depth_[:,:,0] = depth[:,:].T
        depth_[:,:,1] = depth[:,:].T
        depth_[:,:,2] = depth[:,:].T
        
        depth_ = depth_/4.0
        #plt.imshow(depth_)
        #plt.show()
        
        norm_depth = np.zeros(np.shape(depth))
        norm_depth = cv2.normalize(depth_,  norm_depth, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(SAVE_PATH_DEPTH + "depth_" +str(index)+ ".png", norm_depth)
        
        print("Saved RGB and depth image! Index: ", index)
        index = index + 1;

start_parse()