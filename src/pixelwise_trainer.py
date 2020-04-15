# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:26:01 2020

@author: delgallegon
"""
from model import warping_cnn_parallel
from model import warping_cnn_single
from model import resnet_cnn
from model import densenet_cnn
from model import pixelwise_cnn
from loaders import torch_image_loader as loader
from utils import generate_misaligned as gm
import global_vars as gv
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torch.nn as nn
import warping_trainer
from visualizers import gradcam
from visualizers import vanilla_activation

class PixelwiseTrainer(warping_trainer.WarpingTrainer):
    
    def __init__(self, name, gpu_device, writer, lr = 0.05, weight_decay = 0.0):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.writer = writer
        self.visualized = False
        self.model = [0]
        self.optimizers = [0]
        self.model_length = 1
        self.model[0] = pixelwise_cnn.PixelwiseCNN()
        self.model[0].to(self.gpu_device)
        self.optimizers[0] = optim.Adam(self.model[0].parameters(), lr = self.lr, weight_decay = weight_decay)
        
        self.weight_penalty = 100.0
    
    def singular_loss(self, pred, target):
        mse_loss = torch.nn.MSELoss()
        cross_entropy_loss = torch.nn.BCELoss()
        #print("Pred min: ", np.ndarray.min(pred.cpu().data.numpy()), " Target min: ", np.ndarray.min(target.cpu().data.numpy()))
        #print("Pred max: ", np.ndarray.max(pred.cpu().data.numpy()), " Target max: ", np.ndarray.max(target.cpu().data.numpy()))
        #return (mse_loss(pred, target) * self.weight_penalty) + cross_entropy_loss(pred, target)
        return cross_entropy_loss(pred, target)

    def visualize_activation(self, layers, input, target):
        fig, ax = plt.subplots(self.model_length * len(layers))
        fig.set_size_inches(12, self.model_length * len(layers) * 6)
        fig.suptitle("Activation regions")
        
        index = 0;
        for j in range(len(layers)):
            visualizer = gradcam.GradCam(self.model[0], target_layer=layers[j])
            cam = visualizer.generate_cam(input, target, self.weight_penalty)
            cam = np.moveaxis(cam, -1, 0)
            cam = np.moveaxis(cam, -1, 0) #for properly displaying image in matplotlib
            ax[index].imshow(cam)
            index = index + 1
                
        plt.savefig(gv.SAVE_PATH_FIGURES + "/activation_latest.png", bbox_inches='tight', pad_inches=0)
        plt.show()
    
    def train(self, warp, rgb):
        warp_gpu = warp.to(self.gpu_device)
        rgb_gpu = rgb.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        rgb_img = rgb[0,:,:,:].numpy()
        rgb_img = np.moveaxis(rgb_img, -1, 0)
        rgb_img = np.moveaxis(rgb_img, -1, 0) #for properly displaying image in matplotlib
        
        self.batch_loss = 0.0
        self.model[0].train();
        pred = self.model[0](warp_gpu)
        self.optimizers[0].zero_grad()
        loss = self.singular_loss(pred, rgb_gpu)
        self.batch_loss = self.batch_loss + loss.cpu().data
        loss.backward()
        self.optimizers[0].step();    
        
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_rgb_img = rgb_img
        self.last_rgb_tensor = torch.unsqueeze(rgb[0,:,:,:], 0)
        
        if(self.visualized == False):
            self.visualized = True
            #self.visualize_activation(["conv1", "conv2", "conv3", "conv4", "conv5"], warp_gpu, rgb_gpu)
    
    def infer(self, warp, rgb):
        warp_gpu = warp.to(self.gpu_device)
        rgb_gpu = rgb.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        rgb_img = rgb[0,:,:,:].numpy()
        rgb_img = np.moveaxis(rgb_img, -1, 0)
        rgb_img = np.moveaxis(rgb_img, -1, 0) #for properly displaying image in matplotlib
        
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_rgb_img = rgb_img
        self.last_rgb_tensor = torch.unsqueeze(rgb[0,:,:,:], 0)
        
        with torch.no_grad():
            overall_pred = []
            loss = 0.0
            self.model[0].eval()
            pred = self.model[0](warp_gpu)
            loss = loss + self.singular_loss(pred, rgb_gpu)
            
            #return first prediction
            return pred[0, 0].cpu().numpy(), loss.cpu().data
        
    def get_last_rgb_img(self):
        return self.last_rgb_img
    
    def get_last_rgb_tensor(self):
        return self.last_rgb_tensor   
        
        
        