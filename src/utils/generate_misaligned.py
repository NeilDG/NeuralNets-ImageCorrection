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
from visualizers import warp_data_visualizer as wdv
from os.path import isfile, join

#custom data dir location
IMAGE_RGB_DIR = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_rgb/"
SAVE_PATH_RGB = 'C:/NN_Dataset/warp_rgb_orig/'
SAVE_PATH_WARP = 'C:/NN_Dataset/warp_rgb_mod/'

SAVE_PATH_RGB_VAL = 'C:/NN_Dataset/warp_rgb_orig_val/'
SAVE_PATH_WARP_VAL = 'C:/NN_Dataset/warp_rgb_mod_val/'

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

def perform_warp(img, W1 ,W2, W3, W4, W5, padding = 100):
    #add padding to image to avoid overflow
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    padded_image = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
    value=[0,0,0])
    padded_dim = np.shape(padded_image)

#    x_disp = rand.randint(-5, 5) * W1
#    y_disp = rand.randint(-5, 5) * W1
#    both_disp = rand.randint(-5, 5) * W1
#    
#    second_disp_x = rand.randint(-5, 5) * W1
#    second_disp_y = rand.randint(-5, 5) * W1

    pts1 = np.float32([[0,0],[x_dim,0],[0,y_dim], [x_dim, y_dim]])
    #pts2 = np.float32([[0,0],[x_dim + x_disp,second_disp_x],[second_disp_y,y_dim + y_disp], [x_dim + both_disp, y_dim + both_disp]])
    pts2 = np.float32([[0,0],[x_dim,0],[0,y_dim], [x_dim, y_dim]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    #print("Original M: ", M)
    while True:
        M[0,1] = M[0,1] + (np.random.random() * W1) - (np.random.random() * W1)
        M[0,2] = M[0,2] + (np.random.random() * W2) - (np.random.random() * W2)
        M[1,0] = M[1,0] + (np.random.random() * W3) - (np.random.random() * W3)
        M[1,2] = M[1,2] + (np.random.random() * W4) - (np.random.random() * W4)
        M[2,0] = M[2,0] + (np.random.random() * W5) - (np.random.random() * W5)
        M[2,1] = M[2,1] + (np.random.random() * W5) - (np.random.random() * W5)
        result = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]))
        inverse_M = np.linalg.inv(M)
        
        #print("New M: ", M)
        #do not generate extreme inverse values
        if(abs(inverse_M[0,1]) <= W1 * 5 and abs(inverse_M[0,2]) <= W2 * 5 and abs(inverse_M[1,0]) <= W3 * 5 and abs(inverse_M[1,2]) <= W4 * 5):
            break
        else:
            M = cv2.getPerspectiveTransform(pts1, pts2)
    
    return result, M, inverse_M

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

def check_generate_data():
    rgb_list = retrieve_kitti_rgb_list();
    print("Images found: ", np.size(rgb_list))
    
    #test read image
    M0_list = []; M1_list = []; M2_list = []; M3_list = []
    for i in range(100):
        img = cv2.imread(rgb_list[i])
        result, M, inverse_M = perform_warp(img, np.random.rand() * 1.5, np.random.rand() * 1.5, np.random.rand() * 1.5, np.random.rand() * 1.5, 0.0025)
        reverse_img = perform_unwarp(result, inverse_M)    
#        plt.title("Original image"); plt.imshow(img); plt.show()
#        plt.title("Warped image"); plt.imshow(result); plt.show()
#        plt.title("Recovered image"); plt.imshow(reverse_img); plt.show()
#        difference = img - reverse_img
#        plt.title("Image difference between orig and recovered"); plt.imshow(difference); plt.show()
        
        #print("Inverse Matrix: ", inverse_M)
        M0_list.append(inverse_M[0,1])
        M1_list.append(inverse_M[0,2])
        M2_list.append(inverse_M[1,0])
        M3_list.append(inverse_M[1,2])
    
    wdv.visualize_individual_M(M0_list, M1_list, M2_list, M3_list)
    
def generate():
    rgb_list = retrieve_kitti_rgb_list();
    print("Images found: ", np.size(rgb_list))
    
    for i in range(np.size(rgb_list)): 
        img = cv2.imread(rgb_list[i])
        result, M, inverse_M = perform_warp(img, np.random.rand() * 1.5, np.random.rand() * 1.5, np.random.rand() * 1.5, np.random.rand() * 1.5, 0.0025)
        inverse_M = inverse_M
        
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
        
        if(i <= 11195):
            cv2.imwrite(SAVE_PATH_RGB + "orig_" +str(i)+ ".png", img)
            cv2.imwrite(SAVE_PATH_WARP + "warp_" +str(i)+ ".png", result)
            np.savetxt(SAVE_PATH_WARP + "warp_" +str(i)+ ".txt", inverse_M)
            if (i % 200 == 0):
                print("Successfully generated transformed image " ,i, ". Saved as train.")
        else:
            cv2.imwrite(SAVE_PATH_RGB_VAL + "orig_" +str(i)+ ".png", img)
            cv2.imwrite(SAVE_PATH_WARP_VAL + "warp_" +str(i)+ ".png", result)
            np.savetxt(SAVE_PATH_WARP_VAL + "warp_" +str(i)+ ".txt", inverse_M)
            if (i % 200 == 0):
                print("Successfully generated transformed image " ,i, ". Saved as val.")
        
    print("Finished generating dataset!")

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    #Main call
    check_generate_data()
    generate()