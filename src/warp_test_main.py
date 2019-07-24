# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:35:33 2019

Main starting point for testing how good the network is.
Contains inference operations and visualization of results
@author: delgallegon
"""



from visualizers import warp_data_visualizer as visualizer
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from utils import generate_misaligned as gm
from loaders import torch_image_loader as loader
import modular_trainer as trainer
import warp_train_main as train_main
from utils import tensor_utils


BATCH_SIZE = 32
CNN_VERSION = "cnn_v3.13.1"
OPTIMIZER_KEY = "optimizer"

def visualize_results(warp_img, M_list, ground_truth_M, index, p = 0.03):
    chance_to_save = np.random.rand()
    if(chance_to_save <= p):
        should_save = True
        visualizer.show_transform_image_test("Validation set: Input image", warp_img, M1 = M_list[0], M2 = M_list[1], 
                                        M3 = M_list[2], M4 = M_list[3], M5 = M_list[4], ground_truth_M = ground_truth_M,
                                        should_save = should_save, index = index)
    else:
        should_save = False
    
def start_test(gpu_device):
    model_list = []
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/1', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 1, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/2', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 2, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/3', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 3, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/4', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 5, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/5', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 6, lr = train_main.LR))
    
    #checkpoint loading here
    CHECKPATH = 'tmp/' + CNN_VERSION +'.pt'
    checkpoint = torch.load(CHECKPATH)
    for model in model_list:
        model.load_saved_states(checkpoint[model.get_name()], checkpoint[model.get_name() + OPTIMIZER_KEY])
 
    print("Loaded checkpt ",CHECKPATH)
    
    test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE)
    #perform inference on batches
    overall_index = 0;
    for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
        print("Batch idx: ", batch_idx)
        for index in range(len(warp)):
            model_Ms = [];
            for model in model_list:
                warp_candidate = torch.unsqueeze(warp[index,:,:,:], 0).to(gpu_device)
                reshaped_t = torch.reshape(transform[index], (1, 9)).type('torch.FloatTensor')
                gt_candidate = torch.reshape(reshaped_t[:,model.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_device)
        
                M, loss = model.single_infer(warp_tensor = warp_candidate, ground_truth_tensor = gt_candidate)
                model_Ms.append(M)
            
            #chance visualize and save result
            warp_img = tensor_utils.convert_to_matplotimg(warp, index)
            visualize_results(warp_img = warp_img, M_list = model_Ms, ground_truth_M = transform[index], index = overall_index)
            overall_index = overall_index + 1
           
    
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_test(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
