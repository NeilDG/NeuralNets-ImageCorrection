# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:16:57 2020

For creating of results for paper

@author: delgallegon
"""
from visualizers import warp_data_visualizer as warp_visualizer
import torch
import cv2
import numpy as np
from loaders import torch_image_loader as loader
import warping_trainer
import warp_train_main as train_main
from utils import tensor_utils
import global_vars as gv
from matplotlib import pyplot as plt

BATCH_SIZE = 32
OPTIMIZER_KEY = "optimizer"

def produce_figures(gpu_device):
    #model loading here
    ct = warping_trainer.WarpingTrainer(train_main.CNN_VERSION, gpu_device = gpu_device, writer = None, lr = train_main.LR)
    CHECKPATH = 'D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/src/tmp/' + train_main.CNN_VERSION +'.pt'
    checkpoint = torch.load(CHECKPATH)
    for i in range(ct.model_length):         
        ct.load_saved_states(i,checkpoint[ct.get_name() + str(i)], checkpoint[ct.get_name() + OPTIMIZER_KEY + str(i)])
 
    print("Loaded checkpt ",CHECKPATH)
    
    test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE, num_image_to_load = 2000)
    dataset_mean = np.loadtxt(gv.IMAGE_PATH_PREDICT + "dataset_mean.txt")
    
    for batch_idx, (rgb, warp, transform, path) in enumerate(test_dataset):
        for i in range(np.shape(warp)[0]):
            warp_candidate = torch.unsqueeze(warp[i,:,:,:], 0)
            reshaped_t = torch.reshape(transform[i], (1, 9)).type('torch.FloatTensor')
            M, loss = ct.infer(warp_candidate, reshaped_t)
            
            M = np.insert(M, 2, 0.0)
            M = np.insert(M, 5, 0.0)
            M = np.append(M, 1.0)
            M = np.reshape(M, (3,3))
            
            warp_img = tensor_utils.convert_to_matplotimg(warp, i)
            ground_truth_img = tensor_utils.convert_to_matplotimg(rgb, i)
            homog_img, homography_M = warp_visualizer.warp_perspective_least_squares(warp_img, ground_truth_img)
            
            image_name = path[i].split(".")[0]
            
            rrl_1_path = gv.RRL_RESULTS_PATH[1] + image_name + "_r" + ".png"
            rrl_img_1 = tensor_utils.load_image(rrl_1_path)
            rrl_img_1 = cv2.resize(rrl_img_1, (gv.WARP_W, gv.WARP_H)) #because size has changed for RRL img
            
            rrl_2_path = gv.RRL_RESULTS_PATH[0] + image_name + "_m" + ".jpg"
            rrl_img_2 = tensor_utils.load_image(rrl_2_path)
            
            own_img = cv2.warpPerspective(warp_img, np.linalg.inv(M), (np.shape(warp_img)[1], np.shape(warp_img)[0]),borderValue = (1,1,1))
            
            produce_single_figure(image_name, [warp_img, homog_img, rrl_img_1, rrl_img_2, own_img, ground_truth_img])

# Produces a single figure. Image order matters!
def produce_single_figure(image_name, images):
    fig, ax = plt.subplots(len(images))
    fig.set_size_inches(12, len(images) * 3)
    for i in range(len(images)):
        ax[i].autoscale(True)
        ax[i].imshow(images[i])
        ax[i].set_axis_off()
    
    plt.subplots_adjust(left = 0.06, wspace=0, hspace=0.1)    
    plt.savefig(gv.SAVE_PATH_FIGURES + "/" +image_name+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    produce_figures(device)
    
    
if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
