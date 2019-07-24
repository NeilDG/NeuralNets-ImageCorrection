# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:47:47 2019
Data visualizer for analyzing input data
@author: delgallegon
"""
from loaders import torch_image_loader as loader
import os
import numpy as np
import cv2
import global_vars as gv
from matplotlib import pyplot as plt

#saves predicted transforms inferred by network. Always set start_index = 0 if you want to
#override saved predictions
def save_predicted_transforms(M_list, start_index = 0):
    for i in range(np.shape(M_list)[0]):
        np.savetxt(gv.SAVE_PATH_PREDICT + "warp_" +str(i + start_index)+ ".txt", M_list[i])
        print("Successfully saved predicted M ", str(i + start_index))

def retrieve_predict_warp_list():
    warp_list = [];
    
    for (dirpath, dirnames, filenames) in os.walk(gv.SAVE_PATH_PREDICT):
        for f in filenames:
            if f.endswith(".txt"):
                warp_list.append(os.path.join(dirpath, f))
    
    return warp_list

def show_transform_image_test(title, rgb, M1, M2, M3, M4, M5, ground_truth_M, should_save, index):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    f.set_size_inches(12,10)
    
    ax1.set_title(title)
    ax1.imshow(rgb)
    
    pred_M = np.copy(ground_truth_M)
    pred_M[0,1] = M1
    pred_M[0,2] = M2
    pred_M[1,0] = M3
    pred_M[1,2] = M4
    pred_M[2,0] = M5
    
    result = cv2.warpPerspective(rgb, ground_truth_M.numpy(), (np.shape(rgb)[1], np.shape(rgb)[0]))
    ax2.set_title("Ground truth")
    ax2.imshow(result)
    
    result = cv2.warpPerspective(rgb, pred_M, (np.shape(rgb)[1], np.shape(rgb)[0]))
    ax3.set_title("Predicted warp")
    ax3.imshow(result)
    
    if(should_save):
        plt.savefig(gv.IMAGE_PATH_PREDICT + "/result_"+str(index)+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    print("Predicted M1 val: ", M1, "Actual val: ",ground_truth_M[0,1].numpy())
    print("Predicted M2 val: ", M2, "Actual val: ",ground_truth_M[0,2].numpy())
    print("Predicted M3 val: ", M3, "Actual val: ",ground_truth_M[1,0].numpy())
    print("Predicted M4 val: ", M4, "Actual val: ",ground_truth_M[1,2].numpy())
    print("Predicted M5 val: ", M5, "Actual val: ",ground_truth_M[2,0].numpy())
    
def show_transform_image(title, rgb, M1, M2, M3, M4, M5, ground_truth_M, should_save, current_epoch, save_every_epoch):
    plt.title(title)
    plt.imshow(rgb)
    if(should_save and current_epoch % save_every_epoch == 0):
        plt.savefig(gv.IMAGE_PATH_PREDICT + "/input_epoch_"+str(current_epoch)+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.set_size_inches(12,10)
    
    pred_M = np.copy(ground_truth_M)
    pred_M[0,1] = M1
    pred_M[0,2] = M2
    pred_M[1,0] = M3
    pred_M[1,2] = M4
    pred_M[2,0] = M5
    result = cv2.warpPerspective(rgb, ground_truth_M.numpy(), (np.shape(rgb)[1], np.shape(rgb)[0]))
    
    ax1.set_title("Ground truth")
    ax1.imshow(result)
    
    
    #result = cv2.perspectiveTransform(rgb, ground_truth_M.numpy())
    result = cv2.warpPerspective(rgb, pred_M, (np.shape(rgb)[1], np.shape(rgb)[0]))
    ax2.set_title("Predicted warp")
    ax2.imshow(result)
    
    if(should_save and current_epoch % save_every_epoch == 0):
        plt.savefig(gv.IMAGE_PATH_PREDICT + "/result_epoch_"+str(current_epoch)+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    print("Predicted M1 val: ", M1, "Actual val: ",ground_truth_M[0,1].numpy())
    print("Predicted M2 val: ", M2, "Actual val: ",ground_truth_M[0,2].numpy())
    print("Predicted M3 val: ", M3, "Actual val: ",ground_truth_M[1,0].numpy())
    print("Predicted M4 val: ", M4, "Actual val: ",ground_truth_M[1,2].numpy())
    print("Predicted M5 val: ", M5, "Actual val: ",ground_truth_M[2,0].numpy()) 


def visualize_individual_M(M0, M1, M2, M3):
    x = np.random.rand(np.shape(M0)[0])

    plt.scatter(x, M0, color = 'g', label = "M0")
    plt.scatter(x, M1, color = 'r', label = "M1")
    plt.scatter(x, M2, color = 'b', label = "M2")
    plt.scatter(x, M3, color = 'y', label = "M3")
    
    plt.legend()
    plt.title("Distribution of generated M elements")
    plt.show()
    
def visualize_transform_M(M_list, label, color = 'g'):
    #print("Norm of predicted vs actual T")
    X = list(range(0, np.shape(M_list)[0]))
    Y = []
    for i in range(np.shape(M_list)[0]):
        Y.append(np.linalg.norm(M_list[i]))
    
    plt.scatter(X, Y, color = color, label = label)
    plt.legend()

def visualize_predict_M(baseline_M, predicted_M_list):
    for i in range(np.shape(predicted_M_list)[0]):
        plt.scatter(i, np.linalg.norm(predicted_M_list[i]), color = 'r')
    
    
    plt.title("Distribution of norm ground-truth T vs predicted T")
    plt.show()
    
    for i in range(1, np.shape(predicted_M_list)[0]):
        diff = abs(predicted_M_list[i] - predicted_M_list[i - 1])
        plt.scatter(i - 1, np.linalg.norm(diff), color = 'b')
    
    plt.title("Differences of predicted T")
    plt.show()

def visualize_input_data(warp_list):
    print("Input RGB distribution via norm")
    for i in range(np.shape(warp_list)[0]):
        plt.scatter(i, np.linalg.norm(warp_list[i][0,:,:]), color = 'b')
        plt.scatter(i, np.linalg.norm(warp_list[i][1,:,:]), color = 'g')
        plt.scatter(i, np.linalg.norm(warp_list[i][2,:,:]), color = 'r')
    plt.show()
    
def main():
    all_transforms = []
    predict_transforms = []
    warp_list = []
    baseline_M = None
    predict_list_files = retrieve_predict_warp_list()
    for pfile in predict_list_files:
        predict_transforms.append(np.loadtxt(pfile))
    
    for batch_idx, (rgb, warp, transform) in enumerate(loader.load_dataset(batch_size = 64)):
        for t in transform:
            baseline_M = t.numpy()
            all_transforms.append(t.numpy())
        
        for warp_img in warp:
            warp_list.append(warp_img.numpy())
       
        visualize_transform_M(all_transforms, color = 'g', label = "training set")
        all_transforms.clear()
        all_transforms = []
        
        if(batch_idx % 500 == 0):
            break
        
    for batch_idx, (rgb, warp, transform) in enumerate(loader.load_test_dataset(batch_size = 64)):
        for t in transform:
            baseline_M = t.numpy()
            all_transforms.append(t.numpy())
        
        for warp_img in warp:
            warp_list.append(warp_img.numpy())
       
        visualize_transform_M(all_transforms, color = 'b', label = "test set")
        all_transforms.clear()
        all_transforms = []
        
        if(batch_idx % 500 == 0):
            break
    
    visualize_predict_M(baseline_M, predict_transforms)
    
    #visualize_input_data(warp_list)
    
if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()

