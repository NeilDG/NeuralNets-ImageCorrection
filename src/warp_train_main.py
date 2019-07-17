# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:16:15 2019

Main starting point for warp image training
@author: delgallegon
"""


from visualizers import warp_data_visualizer as visualizer
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import modular_trainer as trainer

LR = 0.0001
num_epochs = 500
BATCH_SIZE = 40
CNN_VERSION = "cnn_v3.09"

def start_train(gpu_device):
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')
    
    model_list = []
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/1', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/2', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/3', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/4', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    
    #checkpoint loading here
    CHECKPATH = 'tmp/' + CNN_VERSION +'.pt'
    #placeholder
    
    for epoch in range(num_epochs):
        
        #train
        model_list[0].train(gt_index = 0, current_epoch = epoch)
        model_list[1].train(gt_index = 1, current_epoch = epoch)
        model_list[2].train(gt_index = 3, current_epoch = epoch)
        model_list[3].train(gt_index = 4, current_epoch = epoch)
        
        #perform inference
        warp_img = model_list[-1].get_last_warp_img()
        warp_tensor = model_list[-1].get_last_warp_tensor()
        ground_truth_M = model_list[-1].get_last_transform()
        plt.title("Training set preview: Input image")
        plt.imshow(warp_img)
        
        M1 = model_list[0].infer(warp_tensor = warp_tensor)
        M2 = model_list[1].infer(warp_tensor = warp_tensor)
        M3 = model_list[2].infer(warp_tensor = warp_tensor)
        M4 = model_list[3].infer(warp_tensor = warp_tensor)
        
        visualizer.show_transform_image(warp_img, M1 = M1, M2 = M2, M3 = M3, M4 = M4, ground_truth_M = ground_truth_M)

    
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_train(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()

