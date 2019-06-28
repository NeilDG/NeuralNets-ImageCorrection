# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:47:47 2019
Data visualizer for analyzing input data
@author: delgallegon
"""
from model import warp_cnn
from model import torch_image_loader as loader
from model import torch_image_dataset as image_dataset
from utils import generate_misaligned as gm
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import cv2
import global_vars as gv
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision import transforms

def visualize_transform_M(M_list):
    #print("Norm of predicted vs actual T")
    
    for i in range(np.shape(M_list)[0]):
        plt.scatter(i, np.linalg.norm(M_list[i]), color = 'g')

def visualize_predict_M(baseline_M, predicted_M_list):
    for i in range(np.shape(predicted_M_list)[0]):
        predict_M = np.copy(baseline_M)
        predict_M[1] = predicted_M_list[i]
        plt.scatter(i, np.linalg.norm(predict_M), color = 'r')
    
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
    predict_list_files = gm.retrieve_predict_warp_list()
    for pfile in predict_list_files:
        predict_transforms.append(np.loadtxt(pfile))
    
    for batch_idx, (rgb, warp, transform) in enumerate(loader.load_dataset(batch_size = 64)):
        for t in transform:
            baseline_M = t.numpy()
            all_transforms.append(t.numpy())
        
        for warp_img in warp:
            warp_list.append(warp_img.numpy())
       
        visualize_transform_M(all_transforms)
        all_transforms.clear()
        all_transforms = []
        
        if(batch_idx % 500 == 0):
            break
    
    visualize_predict_M(baseline_M, predict_transforms)
    
    #visualize_input_data(warp_list)
    
if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()

