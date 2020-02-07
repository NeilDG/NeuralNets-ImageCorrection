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
    
    #print("Original M: ", M)
    M[0,1] = np.random.uniform(-0.00075, 0.00075);
    #M[0,2] = np.random.uniform(-10, 10);
    M[1,0] = np.random.uniform(-0.00075, 0.00075);
    #M[1,2] = M[1,2] + (np.random.random() * W4) - (np.random.random() * W4)
    M[2,0] = np.random.uniform(-0.00075, 0.00075)
    M[2,1] = np.random.uniform(-0.00075, 0.00075)
    result = cv2.warpPerspective(padded_image, M, (padded_dim[1], padded_dim[0]), borderValue = (0,0,0))
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
    #remove padding first before unwarping
    # dim = np.shape(img)
    # initial_result = cv2.warpPerspective(img, inverse_M, (dim[1], dim[0]), borderValue = (255,255,255))
    
    # x_dim = np.shape(img)[0]; y_dim = np.shape(img)[1];
    # upper_x = x_dim - padding_deduct
    # upper_y = y_dim - padding_deduct
    # roi_image = initial_result[padding_deduct:upper_x, padding_deduct:upper_y]
    padded_image = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
                                          value=[255,255,255])
    padded_dim = np.shape(padded_image)
    roi_image = cv2.warpPerspective(img, inverse_M, (padded_dim[1], padded_dim[0]), borderValue = (255,255,255))
    roi_image = cv2.resize(roi_image, (gv.WARP_W, gv.WARP_H)) 
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

def generate_single_data(img):
    
    x_ratio = 0.0; y_ratio = 0.0; z_ratio = 0.0
    while(x_ratio < 0.7 and y_ratio < 0.7 and z_ratio < 0.7):
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
        
        #print("SHAPES. Cropped: " ,np.shape(crop), " Resized: ", np.shape(crop_img))
        
        #compute z ratio
        x_ratio = (np.shape(crop)[0] + gv.PADDING_CONSTANT) / np.shape(crop_img)[0]
        y_ratio = (np.shape(crop)[1] + gv.PADDING_CONSTANT) / np.shape(crop_img)[1]
        z_ratio = ((x_ratio * 0.5) + (y_ratio * 0.5)) / 1.0
        print("Ratio X:", x_ratio, " Ratio Y: ", y_ratio, "Ratio Z: ", z_ratio)
        
        M[0,0] = x_ratio; M[1,1] = y_ratio; M[2,2] = z_ratio
        inverse_M = np.linalg.inv(M)
    
    return result, M, inverse_M, crop_img
    
def check_generate_data():
    rgb_list = retrieve_kitti_rgb_list();
    print("Images found: ", np.size(rgb_list))
    
    #test read image
    M0_list = []; M1_list = []; M2_list = []; M3_list = []; M4_list= []; M5_list = []
    
    for i in range(5):
        img = cv2.imread(rgb_list[i])
        result, M, inverse_M, crop_img = generate_single_data(img)      
        reverse_img = perform_unwarp(crop_img, inverse_M)

        plt.title("Original image"); plt.imshow(img); plt.show()
        plt.title("Warped image (Original Res)"); plt.imshow(result); plt.show()
        plt.title("Warped image"); plt.imshow(crop_img); plt.show()
        plt.title("Recovered image"); plt.imshow(reverse_img); plt.show()
        #difference = img - reverse_img
        #plt.title("Image difference between orig and recovered"); plt.imshow(difference); plt.show()
        
        M0_list.append(M[0,1])
        M1_list.append(M[0,2])
        M2_list.append(M[1,0])
        M3_list.append(M[1,2])
        M4_list.append(M[2,0])
        M5_list.append(M[2,1])
    
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

def generate(repeat = 1):
    rgb_list = retrieve_kitti_rgb_list();
    num_images = np.size(rgb_list)
    train_split = int(num_images * repeat * 0.95)
    print("Images found: ", num_images * repeat, "Train split: ", train_split)
    
    index = 0
    for j in range(repeat):
        for i in range(np.size(rgb_list)): 
            img = cv2.imread(rgb_list[i])
            
            result, M, inverse_M, crop_img = generate_single_data(img)      
            reverse_img = perform_unwarp(crop_img, inverse_M)
            
            img = cv2.resize(img, (gv.IMAGE_W, gv.IMAGE_H)) 
            img = cv2.copyMakeBorder(img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT,
                                              value=[255,255,255])
            result = cv2.resize(result, (gv.WARP_W, gv.WARP_H))
            orig_result = perform_padded_warp(img, M)
            
            # f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
            # f.set_size_inches(20,25)
            
            # ax1.imshow(img)
            # ax2.imshow(orig_result)
            # ax3.imshow(crop_img)
            # ax4.imshow(reverse_img)
            # plt.show()
            
            if(index < (train_split + index)):
                cv2.imwrite(gv.SAVE_PATH_RGB + "orig_" +str(index)+ ".png", orig_result)
                cv2.imwrite(gv.SAVE_PATH_RGB_GT + "crop_" +str(index)+ ".png", reverse_img)
                cv2.imwrite(gv.SAVE_PATH_WARP + "warp_" +str(index)+ ".png", crop_img)
                np.savetxt(gv.SAVE_PATH_WARP + "warp_" +str(index)+ ".txt", M)
                if (i % 200 == 0):
                    print("Successfully generated transformed image " ,str(index), ". Saved as train.")
            else:
                cv2.imwrite(gv.SAVE_PATH_RGB_VAL + "orig_" +str(index)+ ".png", orig_result)
                cv2.imwrite(gv.SAVE_PATH_RGB_GT_VAL + "crop_" +str(index)+ ".png", reverse_img)
                cv2.imwrite(gv.SAVE_PATH_WARP_VAL + "warp_" +str(index)+ ".png", crop_img)
                np.savetxt(gv.SAVE_PATH_WARP_VAL + "warp_" +str(index)+ ".txt", M)
                if (i % 200 == 0):
                    print("Successfully generated transformed image " ,str(index), ". Saved as val.")
            
            index = index + 1
            
        print("Finished generating dataset!")

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    #Main call
    #batch_iterative_warp()
    #check_generate_data() 
    generate(5)