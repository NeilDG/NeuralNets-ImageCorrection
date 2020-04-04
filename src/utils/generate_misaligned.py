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
    
def perform_warp(img):
    #add padding to image to avoid overflow
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    padded_image = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
    value=[0,0,0])
    padded_dim = np.shape(padded_image)

    pts1 = np.float32([[0,0],[x_dim,0],[0,y_dim], [x_dim, y_dim]])
    pts2 = np.float32([[0,0],[x_dim,0],[0,y_dim], [x_dim, y_dim]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    M[0,0] = np.random.uniform(0.8, 1.2)
    M[1,1] = np.random.uniform(0.8, 1.2)
    M[0,1] = np.random.uniform(-0.009, 0.009)
    M[1,0] = np.random.uniform(-0.009, 0.009)
    M[2,0] = np.random.uniform(-0.00075, 0.00075)
    M[2,1] = np.random.uniform(-0.00075, 0.00075)
    result = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]))
    inverse_M = np.linalg.inv(M)
    
    return result, M, inverse_M

def perform_padded_warp(img, M):
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    padded_image = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
    value=[255,255,255])
    padded_dim = np.shape(padded_image)
    result = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]), borderValue = (255,255,255))
    
    return result

#for debugging and analysis
def perform_iterative_warp(img):
    #add padding to image to avoid overflow
    x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    padded_image = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
    value=[0,0,0])
    padded_dim = np.shape(padded_image)
    pts1 = np.float32([[0,0],[x_dim,0],[0,y_dim], [x_dim, y_dim]])
    pts2 = np.float32([[0,0],[x_dim,0],[0,y_dim], [x_dim, y_dim]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print("Original M: ", M)

    upper_bounds = M
    M[1,0] = 0
    result_img = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]), borderValue = (0,0,0))
    plt.imshow(result_img)
    plt.show()  
    
    for i in range(30):
        M[1,0] = M[1,0] + 0.0333
        print(M[1,0])
        result_img = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]), borderValue = (0,0,0))
        plt.imshow(result_img)
        plt.show()  
    
def perform_unwarp(img, inverse_M):
    # padded_image = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
    #                                       value=[255,255,255])
    # padded_dim = np.shape(padded_image)
    # roi_image = cv2.warpPerspective(img, inverse_M, (padded_dim[1], padded_dim[0]), borderValue = (255,255,255))
    # roi_image = cv2.resize(roi_image, (gv.WARP_W, gv.WARP_H)) 

    padded_dim = np.shape(img)
    roi_image = cv2.warpPerspective(img, inverse_M, (padded_dim[1], padded_dim[0]), borderValue = (255,255,255))
    #roi_image = cv2.resize(roi_image, (gv.WARP_W, gv.WARP_H)) 
    return roi_image

def resize_by_border_filling(img):
    # border_v = 0;
    # border_h = 0;
    # if (gv.WARP_H/gv.WARP_W) >= (img.shape[0]/img.shape[1]):
    #     border_v = int((((gv.WARP_H/gv.WARP_W)*img.shape[1])-img.shape[0])/2)
    # else:
    #     border_h = int((((gv.WARP_H/gv.WARP_W)*img.shape[0])-img.shape[1])/2)
    
    # if(border_v < 0 or border_h < 0):
    #     print("Border_V: ", border_v, "Border_H: ", border_h)
    # img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0, value=[255,255,255])
    
    height, width = img.shape[:2]
    blank_image = np.zeros((gv.WARP_H,gv.WARP_W,3), np.uint8)
    blank_image[:,:] = (255,255,255)
    
    l_img = blank_image.copy()
    
    x_offset = int((gv.WARP_W - width)/2)
    y_offset = int((gv.WARP_H - height)/2)
    l_img[y_offset:y_offset+height, x_offset:x_offset+width]= img.copy()
    return l_img
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
    
    crop_img = cv2.resize(crop, (gv.WARP_W, gv.WARP_H)) 
    border_fill = resize_by_border_filling(crop)
    return crop_img, border_fill

def batch_iterative_warp():
    rgb_list = retrieve_kitti_rgb_list();
    print("Images found: ", np.size(rgb_list))
    
    for i in range(1):
        img = cv2.imread(rgb_list[i])
        perform_iterative_warp(img)

def generate_single_data(image_name, img):
    
    x_ratio = 0.0; y_ratio = 0.0; z_ratio = 1.0; kp_count = 0; limit = 0; HARD_LIMIT = 100
    #x_ratio < 0.8 or y_ratio < 0.8 or z_ratio < 0.65 or kp_count < 600
    
    result = img
    M = np.zeros((3,3)); M[0,0] = 1.0; M[1,1] = 1.0; M[2,2] = 1.0
    inverse_M = np.linalg.inv(M)
    
    while(limit < HARD_LIMIT and (x_ratio < 0.4 or y_ratio < 0.4 or z_ratio < 0.35 or kp_count < 500)):
        limit = limit + 1
        result, M, inverse_M = perform_warp(img)
        threshold = 1
        gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
        
        _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = result[y:y+h,x:x+w]
        crop = polish_border(crop)
        crop_img = cv2.resize(crop, (gv.WARP_W, gv.WARP_H)) 
        
        x_ratio = (w) / gv.WARP_W
        y_ratio = (h) / gv.WARP_H
        z_ratio = (w * h) / (gv.WARP_W * gv.WARP_H);
                
        #Initiate ORB detector for image verification
        orb = cv2.ORB_create(700)
        kp = orb.detect(crop_img,None)
        kp, des = orb.compute(crop_img, kp)
        kp_count = np.shape(kp)[0]
        print("Name: ", image_name, x_ratio, y_ratio, z_ratio, kp_count)
    
    if(limit < HARD_LIMIT):
        return result, M, inverse_M, crop_img
    else:
        return None, None, None, None
    
def check_generate_data():
    rgb_list = retrieve_kitti_rgb_list();
    print("Images found: ", np.size(rgb_list))
    
    #test read image
    M_list = []
    
    for i in range(5):
        img = cv2.imread(rgb_list[i])
        image_name = rgb_list[i].split("/")[2].split(".")[0]
        result, M, inverse_M, crop_img = generate_single_data(image_name, img)      
        reverse_img = perform_unwarp(crop_img, inverse_M)

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        ax1.axis("off"); ax2.axis("off"); ax3.axis("off"); ax4.axis("off")
        f.set_size_inches(12,15)
        
        #ax1.set_title("Warped image (original res)"); 
        ax1.imshow(result);
        #ax2.set_title("Warped image") 
        ax2.imshow(crop_img);
        #ax3.set_title("Recovered image")
        ax3.imshow(reverse_img);
        img = cv2.resize(img, (gv.WARP_W, gv.WARP_H))
        #ax4.set_title("Original image") 
        ax4.imshow(img); plt.show()
        
        difference = np.abs(img - reverse_img)
        plt.title("Image difference between orig and recovered"); plt.imshow(difference); plt.show()
        
        #print(M)
        M_list.append(M[0,0])
        M_list.append(M[0,1])
        M_list.append(M[1,0])
        M_list.append(M[1,1])
        M_list.append(M[2,0])
        M_list.append(M[2,1])
        
    wdv.visualize_M_list(M_list)

def generate_unseen_samples(repeat = 1, offset = 0):
    rgb_list = retrieve_unseen_list()
    #rgb_list = retrieve_kitti_rgb_list()
    num_images = np.size(rgb_list)
    print("Images found: ", num_images * repeat)
    
    index = 0
    offset_index = index + offset
    for j in range(repeat):
        for i in range(np.size(rgb_list)):
            img = cv2.imread(rgb_list[i])
            image_name = rgb_list[i].split("/")[2].split(".")[0]
            result, M, inverse_M, crop_img = generate_single_data(image_name, img) 
            if(result is None):
                continue
            
            reverse_img = perform_unwarp(crop_img, inverse_M)
            
            img = cv2.resize(img, (gv.IMAGE_W, gv.IMAGE_H)) 
            img = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
                                              value=[255,255,255])
            result = cv2.resize(result, (gv.WARP_W, gv.WARP_H))
            #orig_result = perform_padded_warp(img, M)
            
            # f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
            # f.set_size_inches(20,25)
            
            # ax1.imshow(img)
            # ax2.imshow(orig_result)
            # ax3.imshow(crop_img)
            # ax4.imshow(reverse_img)
            # plt.show()
            
            cv2.imwrite(gv.SAVE_PATH_UNSEEN_DATA_RGB_GT + "crop_" +str(offset_index)+ ".png", reverse_img)
            cv2.imwrite(gv.SAVE_PATH_UNSEEN_DATA_WARP + "warp_" +str(offset_index)+ ".png", crop_img)
            np.savetxt(gv.SAVE_PATH_UNSEEN_DATA_WARP + "warp_" +str(offset_index)+ ".txt", M, fmt = "%.8f")
            if (i % 200 == 0):
                print("Successfully generated transformed image " ,str(offset_index), ". Saved as train.")
            
            index = index + 1
            offset_index = index + offset
        print("Finished generating unseen dataset!")

def refine_data(orig_img, result, M, inverse_M, reverse_img, threshold):
   num_edge = wdv.count_edge_from_img(reverse_img)
   while(num_edge < threshold):
       print("Edge count not enough! Regenerating! Edge count is only ", num_edge)
       result, M, inverse_M = perform_warp(orig_img, np.random.rand() * 0.005, np.random.rand() * 0.005, 
                                            np.random.rand() * 0.005, np.random.rand() * 0.005, 
                                            WARP_MULT)
       result = remove_border_and_resize(result, 1)
       reverse_img = perform_unwarp(result, inverse_M)
       num_edge = wdv.count_edge_from_img(reverse_img)
   return result, M, inverse_M, reverse_img

def generate(repeat = 1, offset = 0):
    rgb_list = retrieve_kitti_rgb_list();
    num_images = np.size(rgb_list)
    train_split = int(num_images * repeat * 0.95)
    print("Images found: ", num_images * repeat, "Train split: ", train_split)
    
    index = 0
    offset_index = index + offset
    for j in range(repeat):
        for i in range(np.size(rgb_list)): 
            img = cv2.imread(rgb_list[i])
            
            result, M, inverse_M, crop_img = generate_single_data(img)      
            reverse_img = perform_unwarp(crop_img, inverse_M)
            
            img = cv2.resize(img, (gv.IMAGE_W, gv.IMAGE_H)) 
            img = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
                                              value=[255,255,255])
            result = cv2.resize(result, (gv.WARP_W, gv.WARP_H))
            #orig_result = perform_padded_warp(img, M)
            
            # f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
            # f.set_size_inches(20,25)
            
            # ax1.imshow(img)
            # ax2.imshow(orig_result)
            # ax3.imshow(crop_img)
            # ax4.imshow(reverse_img)
            # plt.show()
            
            if(index < train_split):
                cv2.imwrite(gv.SAVE_PATH_RGB_GT + "crop_" +str(offset_index)+ ".png", reverse_img)
                cv2.imwrite(gv.SAVE_PATH_WARP + "warp_" +str(offset_index)+ ".png", crop_img)
                np.savetxt(gv.SAVE_PATH_WARP + "warp_" +str(offset_index)+ ".txt", M, fmt = "%.8f")
                if (i % 200 == 0):
                    print("Successfully generated transformed image " ,str(offset_index), ". Saved as train.")
            else:
                cv2.imwrite(gv.SAVE_PATH_RGB_GT_VAL + "crop_" +str(offset_index)+ ".png", reverse_img)
                cv2.imwrite(gv.SAVE_PATH_WARP_VAL + "warp_" +str(offset_index)+ ".png", crop_img)
                np.savetxt(gv.SAVE_PATH_WARP_VAL + "warp_" +str(offset_index)+ ".txt", M, fmt = "%.8f")
                if (i % 200 == 0):
                    print("Successfully generated transformed image " ,str(offset_index), ". Saved as val.")
            
            index = index + 1
            offset_index = index + offset
        print("Finished generating dataset!")

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    #Main call
    #batch_iterative_warp()
    #check_generate_data() 
    #generate(2, 0)
    generate_unseen_samples(1)