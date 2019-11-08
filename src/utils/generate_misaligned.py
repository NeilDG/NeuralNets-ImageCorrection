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

WARP_MULT = 0.001;

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]

def retrieve_kitti_rgb_list():
    rgb_list = [];
    
    for (dirpath, dirnames, filenames) in os.walk(gv.IMAGE_RGB_DIR):
        for d in dirnames:
            if d.endswith("image_02"):
                for (dirpath, dirnames, filenames) in os.walk(dirpath + "/" + d):
                    for f in filenames:
                        if f.endswith(".png"):
                            rgb_list.append(os.path.join(dirpath, f))
    
    return rgb_list

def retrieve_unseen_list():
    rgb_list = [];
    
    for (dirpath, dirnames, filenames) in os.walk(gv.SAVE_PATH_UNSEEN_DATA):
       for f in filenames:
           if f.endswith(".jpg"):
               rgb_list.append(os.path.join(dirpath, f))
             
    
    return rgb_list
    
    return rgb_list
def perform_warp(img, W1 ,W2, W3, W4, W5):
    #add padding to image to avoid overflow
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    padded_image = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
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
        result = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]), borderValue = (0,0,0))
        inverse_M = np.linalg.inv(M)
        
        #print("New M: ", M)
        #do not generate extreme inverse values
#        if(abs(inverse_M[0,1]) <= W1 * 5 and abs(inverse_M[0,2]) <= W2 * 5 and abs(inverse_M[1,0]) <= W3 * 5 and abs(inverse_M[1,2]) <= W4 * 5):
#            break
#        else:
#            M = cv2.getPerspectiveTransform(pts1, pts2)
        
        break;
    
    return result, M, inverse_M

def perform_unwarp(img, inverse_M, padding_deduct = 100):
    #remove padding first before unwarping
    dim = np.shape(img)
    initial_result = cv2.warpPerspective(img, inverse_M, (dim[1], dim[0]), borderValue = (255,255,255))
    
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    upper_x = x_dim - padding_deduct
    upper_y = y_dim - padding_deduct
    roi_image = initial_result[padding_deduct:upper_x, padding_deduct:upper_y]
    #roi_dim = np.shape(roi_image)
    
    return roi_image

"""
Polishes the image by further removing the border via non-zero checking
"""
def polish_border(warp_img, zero_threshold = 100, cut = 10):
    gray = cv2.cvtColor(warp_img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
    h,w = np.shape(thresh)
    num_zeros = (h * w) - cv2.countNonZero(thresh)
    
    box = [0,h,0,w]
    crop = warp_img
    
    while(num_zeros > zero_threshold):
       box[0] = box[0] + cut; box[1] = box[1] - cut
       box[2] = box[2] + cut; box[3] = box[3] - cut
       old = crop
       crop = warp_img[box[0]: box[1], box[2]: box[3]]
       
       gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
       _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
       
       if(len(np.shape(thresh)) != 0):
           h,w = np.shape(thresh)
           num_zeros = (h * w) - cv2.countNonZero(thresh)
       else:
           num_zeros = 0
           crop = old
           #print("Can no longer trim!")
       
    return crop
    
def remove_border_and_resize(warp_img, threshold):
    
    gray = cv2.cvtColor(warp_img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
    #plt.imshow(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)); plt.show()
    
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    crop = warp_img[y:y+h,x:x+w]

    crop = polish_border(crop)
    crop = cv2.resize(crop, (gv.WARP_W, gv.WARP_H)) 
    return crop

def check_generate_data():
    rgb_list = retrieve_kitti_rgb_list();
    print("Images found: ", np.size(rgb_list))
    
    #test read image
    M0_list = []; M1_list = []; M2_list = []; M3_list = []; M4_list= []; M5_list = []
    
    for i in range(20):
        img = cv2.imread(rgb_list[i])
        result, M, inverse_M = perform_warp(img, np.random.rand() * 0.005, np.random.rand() * 0.005, 
                                            np.random.rand() * 0.005, np.random.rand() * 0.005, 
                                            WARP_MULT)
        result = remove_border_and_resize(result, 1)
        reverse_img = perform_unwarp(result, inverse_M)  
        plt.title("Original image"); plt.imshow(img); plt.show()
        plt.title("Warped image"); plt.imshow(result); plt.show()
        plt.title("Recovered image"); plt.imshow(reverse_img); plt.show()
        difference = img - reverse_img
        plt.title("Image difference between orig and recovered"); plt.imshow(difference); plt.show()
        
        M0_list.append(inverse_M[0,1])
        M1_list.append(inverse_M[0,2])
        M2_list.append(inverse_M[1,0])
        M3_list.append(inverse_M[1,2])
        M4_list.append(inverse_M[2,0])
        M5_list.append(inverse_M[2,1])
    
    M_parent_list = [];
    M_parent_list.append(M0_list)
    M_parent_list.append(M1_list)
    M_parent_list.append(M2_list)
    M_parent_list.append(M3_list)
    M_parent_list.append(M4_list)
    M_parent_list.append(M5_list)
    
    wdv.visualize_M_list(M_list = M_parent_list)

def generate_unseen_samples(repeat):
    rgb_list = retrieve_unseen_list();
    print("Unseen images found: ", rgb_list)
    count = 0
    for i in range(np.size(rgb_list)): 
        img = cv2.imread(rgb_list[i])
        w,h,a = np.shape(img)
        #crop image
        #center_x = int(np.round(h / 2) - 600); center_y =  int(np.round(w / 2) + 1200)
        #bounds_x =  int(np.round(IMAGE_H / 2)); bounds_y =  int(np.round(IMAGE_W / 2))
        #box = [center_x - bounds_x, center_y - bounds_y, center_x + bounds_x, center_y + bounds_y]
        img = img[0:gv.IMAGE_W, 500::]
        img = cv2.resize(img, (gv.IMAGE_W, gv.IMAGE_H)) 
        
        for j in range(repeat):
            result, M, inverse_M = perform_warp(img, np.random.rand() * WARP_MULT, np.random.rand() * WARP_MULT, np.random.rand() * WARP_MULT, np.random.rand() * WARP_MULT, WARP_MULT)
            inverse_M = inverse_M
            
#            reverse_img = perform_unwarp(result, inverse_M)       
#            plt.imshow(img)
#            plt.show()
#            
#            plt.imshow(reverse_img)
#            plt.show()
#            
#            difference = img - reverse_img
#            plt.imshow(difference)
#            plt.show()
            
            cv2.imwrite(gv.SAVE_PATH_UNSEEN_DATA_RGB + "orig_" +str(count)+ ".png", img)
            cv2.imwrite(gv.SAVE_PATH_UNSEEN_DATA_WARP + "warp_" +str(count)+ ".png", result)
            np.savetxt(gv.SAVE_PATH_UNSEEN_DATA_WARP + "warp_" +str(count)+ ".txt", inverse_M)
            count = count + 1
        
def generate(index_start = 0):
    rgb_list = retrieve_kitti_rgb_list();
    print("Images found: ", np.size(rgb_list))
    
    NO_WARP_CHANCE = 0.05;
    
    for i in range(np.size(rgb_list)): 
        img = cv2.imread(rgb_list[i])
        dice_roll = np.random.rand();
        if(dice_roll < NO_WARP_CHANCE):
            result, M, inverse_M = perform_warp(img,0, 0, 0, 0, 0)
            result = remove_border_and_resize(result, 1)
        else:
            result, M, inverse_M = perform_warp(img, np.random.rand() * 0.005, np.random.rand() * 0.005, 
                                            np.random.rand() * 0.005, np.random.rand() * 0.005, 
                                            WARP_MULT)
            result = remove_border_and_resize(result, 1)
        reverse_img = perform_unwarp(result, inverse_M)       
#        plt.imshow(img)
#        plt.show()
#        
#        plt.imshow(reverse_img)
#        plt.show()
#        
#        difference = img - reverse_img
#        plt.imshow(difference)
#        plt.show()
        
        img = cv2.resize(img, (gv.IMAGE_W, gv.IMAGE_H)) 
        img = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
                                          value=[255,255,255])
        result = cv2.resize(result, (gv.WARP_W, gv.WARP_H))
        
        if(i + index_start <= 11195 + index_start):
            cv2.imwrite(gv.SAVE_PATH_RGB + "orig_" +str(i + index_start)+ ".png", img)
            cv2.imwrite(gv.SAVE_PATH_RGB_CROPPED + "crop_" +str(i + index_start)+ ".png", reverse_img)
            cv2.imwrite(gv.SAVE_PATH_WARP + "warp_" +str(i + index_start)+ ".png", result)
            np.savetxt(gv.SAVE_PATH_WARP + "warp_" +str(i + index_start)+ ".txt", inverse_M)
            if (i % 200 == 0):
                print("Successfully generated transformed image " ,str(i + index_start), ". Saved as train.")
        else:
            cv2.imwrite(gv.SAVE_PATH_RGB_VAL + "orig_" +str(i + index_start)+ ".png", img)
            cv2.imwrite(gv.SAVE_PATH_RGB_CROPPED_VAL + "crop_" +str(i + index_start)+ ".png", reverse_img)
            cv2.imwrite(gv.SAVE_PATH_WARP_VAL + "warp_" +str(i + index_start)+ ".png", result)
            np.savetxt(gv.SAVE_PATH_WARP_VAL + "warp_" +str(i + index_start)+ ".txt", inverse_M)
            if (i % 200 == 0):
                print("Successfully generated transformed image " ,str(i + index_start), ". Saved as val.")
        
    print("Finished generating dataset!")

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    #Main call
    #check_generate_data()
    generate(index_start = 0)
    generate(index_start = 11196)
    #generate_unseen_samples(repeat = 15)