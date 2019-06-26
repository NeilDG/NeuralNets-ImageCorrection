# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:19:29 2019

Code that generates random misalignments through transforms
@author: delgallegon
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import random as rand
import global_vars as gv
from os.path import isfile, join

#custom data dir location
IMAGE_RGB_DIR = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_rgb/"
SAVE_PATH_RGB = 'D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/warp_rgb_orig/'
SAVE_PATH_WARP = 'D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/warp_rgb_mod/'
SAVE_PATH_PREDICT = 'D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/warp_rgb_predict/'

IMAGE_W = 1242; IMAGE_H = 375
WARP_W = 1442; WARP_H = 575

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]

def retrieve_kitti_rgb_list():
    rgb_list = [];
    
    for (dirpath, dirnames, filenames) in os.walk(IMAGE_RGB_DIR):
        for d in dirnames:
            if d.endswith("image_02"):
                for (dirpath, dirnames, filenames) in os.walk(dirpath + "/" + d):
                    for f in filenames:
                        if f.endswith(".png"):
                            rgb_list.append(os.path.join(dirpath, f))
    
    return rgb_list

def retrieve_predict_warp_list():
    warp_list = [];
    
    for (dirpath, dirnames, filenames) in os.walk(SAVE_PATH_PREDICT):
        for f in filenames:
            if f.endswith(".txt"):
                warp_list.append(os.path.join(dirpath, f))
    
    return warp_list

def perform_warp(img, warp_intensity, displacement, padding = 100):
    #add padding to image to avoid overflow
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    padded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
    value=[0,0,0])
    padded_dim = np.shape(padded_image)

    x_disp = rand.randint(-displacement, displacement) * warp_intensity
    y_disp = rand.randint(-displacement, displacement) * warp_intensity
    both_disp = rand.randint(-displacement, displacement) * warp_intensity
    
    second_disp_x = rand.randint(-displacement, displacement) * warp_intensity
    second_disp_y = rand.randint(-displacement, displacement) * warp_intensity

    pts1 = np.float32([[0,0],[x_dim,0],[0,y_dim], [x_dim, y_dim]])
    pts2 = np.float32([[0,0],[x_dim + x_disp,second_disp_x],[second_disp_y,y_dim + y_disp], [x_dim + both_disp, y_dim + both_disp]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]))
    
    return result, M, np.linalg.inv(M)

def perform_unwarp(img, inverse_M, padding_deduct = 100):
    #remove padding first before unwarping
    dim = np.shape(img)
    initial_result = cv2.warpPerspective(img, inverse_M, (dim[1], dim[0]))
    
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    upper_x = x_dim - padding_deduct
    upper_y = y_dim - padding_deduct
    roi_image = initial_result[padding_deduct:upper_x, padding_deduct:upper_y]
    #roi_dim = np.shape(roi_image)
    
    return roi_image

def generate():
    rgb_list = retrieve_kitti_rgb_list();
    
    #test read image
    print("Images found: ", np.size(rgb_list))
    for i in range(np.size(rgb_list)): 
        img = cv2.imread(rgb_list[i])
        result, M, inverse_M = perform_warp(img, 5, 10)
        inverse_M = inverse_M
        #print("Inverse M with constant: ", inverse_M)
        #inverse_M = inverse_M / gv.WARPING_CONSTANT
        #print("Inverse M without constant: ", inverse_M)
#        reverse_img = perform_unwarp(result, inverse_M)       
#        plt.imshow(img)
#        plt.show()
#        
#        plt.imshow(reverse_img)
#        plt.show()
#        
#        difference = img - reverse_img
#        plt.imshow(difference)
#        plt.show()


        img = cv2.resize(img, (IMAGE_W, IMAGE_H)) 
        result = cv2.resize(result, (WARP_W, WARP_H))
        cv2.imwrite(SAVE_PATH_RGB + "orig_" +str(i)+ ".png", img)
        cv2.imwrite(SAVE_PATH_WARP + "warp_" +str(i)+ ".png", result)
        np.savetxt(SAVE_PATH_WARP + "warp_" +str(i)+ ".txt", inverse_M)
        print("Successfully generated transformed image " ,i, ".")

#saves predicted transforms inferred by network. Always set start_index = 0 if you want to
#override saved predictions
def save_predicted_transforms(M_list, start_index = 0):
    for i in range(np.shape(M_list)[0]):
        np.savetxt(SAVE_PATH_PREDICT + "warp_" +str(i + start_index)+ ".txt", M_list[i])
        print("Successfully saved predicted M ", str(i + start_index))

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    #Main call
    generate()