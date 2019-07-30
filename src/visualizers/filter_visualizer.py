# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:03:09 2019
Visualizer class for showing the layers of the CNN
@author: delgallegon
"""
import torch
import torchvision
import numpy as np
import cv2
from matplotlib import pyplot as plt

def visualize_layer(layer, filter_range, resize_scale = 1):
    x,y = np.shape(layer[0,0].data.numpy())
    fig = plt.figure(figsize=(x / 3.0, filter_range * 3))
    #fig = plt.figure()
    
    for i in range(filter_range):
        ax = fig.add_subplot(filter_range, 5, i+1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        
        activation = layer[0,i].data.numpy()
        x,y = np.shape(activation)
        activation_img = cv2.resize(activation, (x * resize_scale, y * resize_scale), interpolation = cv2.INTER_CUBIC)  # scale image up
        #print("Activation shape :", np.shape(activation_img))
        ax.imshow(np.squeeze(activation_img), cmap='gray')
        ax.set_title('%s' % str(i+1))
    
    plt.subplots_adjust(wspace=0, hspace=0.35)
    plt.show()
        

