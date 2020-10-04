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
from visualizers import gradcam
from torchvision import transforms

BATCH_SIZE = 32
OPTIMIZER_KEY = "optimizer"

def infer_single(gpu_device):
    #model loading here
    ct = warping_trainer.WarpingTrainer(train_main.CNN_VERSION, gpu_device = gpu_device, writer = None, lr = train_main.LR)
    CHECKPATH = 'D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/src/tmp/' + train_main.CNN_VERSION +'.pt'
    checkpoint = torch.load(CHECKPATH)
    for i in range(ct.model_length):         
        ct.load_saved_states(i,checkpoint[ct.get_name() + str(i)], checkpoint[ct.get_name() + OPTIMIZER_KEY + str(i)])
    
    INPUT_IMAGE_PATH = "D:/Datasets/NN_Dataset/real_world/samples/4.png"
    print("Successfully loaded model. Image path:" ,INPUT_IMAGE_PATH)
    input_img = cv2.imread(INPUT_IMAGE_PATH)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (gv.WARP_W, gv.WARP_H))
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    input_img_tensor = generic_transform(input_img)
    input_img_tensor = torch.unsqueeze(input_img_tensor, 0)
    
    M = ct.infer_image(input_img_tensor)
    M = np.insert(M, 2, 0.0)
    M = np.insert(M, 5, 0.0)
    M = np.append(M, 1.0)
    M = np.reshape(M, (3,3))
    
    print(M)
    input_img = tensor_utils.convert_to_matplotimg(input_img_tensor, 0)
    own_img = cv2.warpPerspective(input_img, np.linalg.inv(M), (gv.WARP_W - 400, gv.WARP_H - 200),borderValue = (1,1,1))
    
    RESULTS_PATH = "D:/Datasets/NN_Dataset/real_world/"
    
    cv2.imwrite(RESULTS_PATH + "corrected.png", cv2.cvtColor(cv2.convertScaleAbs(own_img, alpha=(255.0)), cv2.COLOR_BGR2RGB))
        
    rrl_1_path = "D:/Datasets/NN_Dataset/real_world/results/4_r.png"
    rrl_img_1 = tensor_utils.load_image(rrl_1_path)
    rrl_img_1 = cv2.resize(rrl_img_1, (gv.WARP_W, gv.WARP_H)) #because size has changed for RRL img
     
    rrl_2_path = "D:/Datasets/NN_Dataset/real_world/results/4_m.jpg"
    rrl_img_2 = tensor_utils.load_image(rrl_2_path)
    rrl_img_2 = cv2.resize(rrl_img_2, (gv.WARP_W, gv.WARP_H)) #because size has changed for RRL img
    
    fig, ax = plt.subplots(4)
    fig.set_size_inches(8, 8)
    for i in range(len(ax)):
        ax[i].autoscale(True)
        ax[i].set_axis_off()
    
    ax[0].imshow(input_img)
    ax[1].imshow(rrl_img_1)
    ax[2].imshow(rrl_img_2)
    ax[3].imshow(own_img)
    plt.subplots_adjust(left = 0.06, wspace=0, hspace=0.1)    
    plt.savefig(RESULTS_PATH + "results.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
def produce_figures(gpu_device):
    #model loading here
    ct = warping_trainer.WarpingTrainer(train_main.CNN_VERSION, gpu_device = gpu_device, writer = None, lr = train_main.LR)
    CHECKPATH = 'D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/src/tmp/' + train_main.CNN_VERSION +'.pt'
    checkpoint = torch.load(CHECKPATH)
    for i in range(ct.model_length):         
        ct.load_saved_states(i,checkpoint[ct.get_name() + str(i)], checkpoint[ct.get_name() + OPTIMIZER_KEY + str(i)])
 
    print("Loaded checkpt ",CHECKPATH)
    #test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE, num_image_to_load = 2000)
    test_dataset = loader.load_unseen_dataset(BATCH_SIZE, 1900)
    dataset_mean = np.loadtxt(gv.IMAGE_PATH_PREDICT + "dataset_mean.txt")
    
    for batch_idx, (rgb, warp, transform, path) in enumerate(test_dataset):
        for i in range(np.shape(warp)[0]):
            warp_candidate = torch.unsqueeze(warp[i,:,:,:], 0)
            reshaped_t = torch.reshape(transform[i], (1, 9)).type('torch.FloatTensor')
            t = [0, 0, 0]
            t[0] = torch.index_select(reshaped_t, 1, torch.tensor([0, 4])).to(gpu_device)
            t[1] = torch.index_select(reshaped_t, 1, torch.tensor([1, 3])).to(gpu_device)
            t[2] = torch.index_select(reshaped_t, 1, torch.tensor([6, 7])).to(gpu_device)
            
            M, loss = ct.infer(warp_candidate, reshaped_t)
            
            M = np.insert(M, 2, 0.0)
            M = np.insert(M, 5, 0.0)
            M = np.append(M, 1.0)
            M = np.reshape(M, (3,3))
            
            warp_img = tensor_utils.convert_to_matplotimg(warp, i)
            warp_img = cv2.resize(warp_img, (gv.PLACES_W, gv.PLACES_H)) #TEMP
            ground_truth_img = tensor_utils.convert_to_matplotimg(rgb, i)
            ground_truth_img = cv2.resize(ground_truth_img, (gv.PLACES_W, gv.PLACES_H)) #TEMP
            homog_img, homography_M = warp_visualizer.warp_perspective_least_squares(warp_img, ground_truth_img)
            homog_img = cv2.resize(homog_img, (gv.PLACES_W, gv.PLACES_H)) #TEMP
            
            image_name = path[i].split(".")[0]
            
            rrl_1_path = gv.RRL_RESULTS_PATH[3] + image_name + "_r" + ".png"
            rrl_img_1 = tensor_utils.load_image(rrl_1_path)
            #rrl_img_1 = cv2.resize(rrl_img_1, (gv.WARP_W, gv.WARP_H)) #because size has changed for RRL img
            rrl_img_1 = cv2.resize(rrl_img_1, (gv.PLACES_W, gv.PLACES_H)) #because size has changed for RRL img
            
            rrl_2_path = gv.RRL_RESULTS_PATH[2] + image_name + "_m" + ".jpg"
            rrl_img_2 = tensor_utils.load_image(rrl_2_path)
            
            if(rrl_img_1 is not None and rrl_img_2 is not None):
                #own_img = cv2.warpPerspective(warp_img, np.linalg.inv(M), (np.shape(warp_img)[1], np.shape(warp_img)[0]),borderValue = (1,1,1))
                #temporary. for unseen dataset
                own_img = cv2.warpPerspective(warp_img, np.linalg.inv(M), (gv.PLACES_W, gv.PLACES_H),borderValue = (1,1,1))
                produce_single_figure(image_name, [warp_img, homog_img, rrl_img_1, rrl_img_2, own_img, ground_truth_img])
            
            #visualize_activation(["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9"], ct, image_name, warp_candidate.to(gpu_device), t)

def visualize_activation(layers, ct, image_name, input, target):
    fig, ax = plt.subplots(len(ct.model), len(layers))
    fig.set_size_inches(50, 10)
    #fig.suptitle("Activation regions")
    
    index = 0;
    #for i in range(len(ct.model)):
    for j in range(len(layers)):
        visualizer = gradcam.GradCam(ct.model[0], target_layer=layers[j])
        cam = visualizer.generate_cam(input, target[0], ct.weight_penalties[0])
        # cam = np.moveaxis(cam, -1, 0)
        # cam = np.moveaxis(cam, -1, 0) #for properly displaying image in matplotlib
        ax[0, j].imshow(cam)
        ax[0, j].set_axis_off()
        index = index + 1
        
        visualizer = gradcam.GradCam(ct.model[0], target_layer=layers[j])
        cam = visualizer.generate_cam(input, target[1], ct.weight_penalties[5])
        # cam = np.moveaxis(cam, -1, 0)
        # cam = np.moveaxis(cam, -1, 0) #for properly displaying image in matplotlib
        ax[1, j].imshow(cam)
        ax[1, j].set_axis_off()
        index = index + 1
        
        visualizer = gradcam.GradCam(ct.model[2], target_layer=layers[j])
        cam = visualizer.generate_cam(input, target[2], ct.weight_penalties[5])
        # cam = np.moveaxis(cam, -1, 0)
        # cam = np.moveaxis(cam, -1, 0) #for properly displaying image in matplotlib
        ax[2, j].imshow(cam)
        ax[2, j].set_axis_off()
        index = index + 1
    
    plt.subplots_adjust(left = 0.06, wspace=0.0, hspace=0.0)            
    plt.savefig(gv.SAVE_PATH_FIGURES + "/" +image_name+ "_activation.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    print("Saved "+image_name+ "_activation.png")
    
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
    
    
    infer_single(device)
    #produce_figures(device)
    
    
if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
