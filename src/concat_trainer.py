# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:41:58 2019

@author: delgallegon
"""

from model import concat_cnn
from model import resnet_cnn
from loaders import torch_image_loader as loader
from utils import generate_misaligned as gm
import global_vars as gv
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torch.nn as nn
from visualizers import gradcam

class ConcatTrainer:
    
    def __init__(self, name, gpu_device, writer, lr = 0.05, weight_decay = 0.0):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.writer = writer
        self.visualized = False
        self.model = [0]
        self.optimizers = [0]
        self.model_length = 1
        for i in range(self.model_length):
            self.model[i] = concat_cnn.ConcatCNN()
            self.model[i].to(self.gpu_device)
            self.optimizers[i] = optim.Adam(self.model[i].parameters(), lr = self.lr, weight_decay = weight_decay)
        
        #self.weight_penalties = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0]
        #self.exp_penalties = [2.0, 1.0, 1.0, 2.0, 1.0, 1.0]
        self.weight_penalties = [1000000.0, 1000000.0,]
        self.exp_penalties = [1.0, 1.0]
    
    def report_new_epoch(self):
        self.visualized = False
        
    def singular_loss(self, pred, target):
        mse_loss = torch.nn.MSELoss(reduction = 'sum')
        
        #0 1 3 4 6 7
        loss_shape = np.shape(pred)[0]
        total_loss = torch.zeros([1], dtype=torch.float64, requires_grad = True, device = self.gpu_device)
        for i in range(len(self.weight_penalties)): 
            total_loss = total_loss + ((mse_loss(pred[:, i], target[:, i]) ** self.exp_penalties[i]) * self.weight_penalties[i])
        
        return total_loss
    
    def visualize_activation(self, layers, input, target):
        fig, ax = plt.subplots(self.model_length * len(layers))
        fig.set_size_inches(12, self.model_length * len(layers) * 6)
        fig.suptitle("Activation regions")
        
        index = 0;
        for i in range(self.model_length):
            for j in range(len(layers)):
                visualizer = gradcam.GradCam(self.model[i], target_layer=layers[j])
                cam = visualizer.generate_cam(input, target[i], self.weight_penalties[i])
                cam = np.moveaxis(cam, -1, 0)
                cam = np.moveaxis(cam, -1, 0) #for properly displaying image in matplotlib
                ax[index].imshow(cam)
                index = index + 1
                
        plt.savefig(gv.SAVE_PATH_FIGURES + "/activation_latest.png", bbox_inches='tight', pad_inches=0)
        plt.show()
    
    def train(self, warp, transform):
        
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
        t = [0]
        t[0] = torch.index_select(reshaped_t, 1, torch.tensor([6, 7])).to(self.gpu_device)
        # t[1] = torch.index_select(reshaped_t, 1, torch.tensor([1])).to(self.gpu_device)
        # t[1] = torch.index_select(reshaped_t, 1, torch.tensor([1])).to(self.gpu_device)
        # t[2] = torch.index_select(reshaped_t, 1, torch.tensor([3])).to(self.gpu_device)
        # t[3] = torch.index_select(reshaped_t, 1, torch.tensor([4])).to(self.gpu_device)
        # t[4] = torch.index_select(reshaped_t, 1, torch.tensor([6])).to(self.gpu_device)
        # t[5] = torch.index_select(reshaped_t, 1, torch.tensor([7])).to(self.gpu_device)
        
        #0 1 2 3 4 5
        self.batch_loss = 0.0
        for i in range(self.model_length):
            self.model[i].train();
            pred = self.model[i](warp_gpu)
            self.optimizers[i].zero_grad()
            loss = self.singular_loss(pred, t[i])
            self.batch_loss = self.batch_loss + loss.cpu().data
            loss.backward()
            self.optimizers[i].step();       
        
        if(self.visualized == False):
            self.visualized = True
            self.visualize_activation(["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8"], warp_gpu, t)
        
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_transform = transform[0]
        self.last_transform_tensor = torch.unsqueeze(reshaped_t[0], 0)
    
    def log_weights(self, current_epoch):
        #log update in weights
        for model in self.model:
            for module_name,module in model.named_modules():
                for name, param in module.named_parameters():
                    if(module_name != ""):
                        #print("Layer added to tensorboard: ", module_name + '/weights/' +name)
                        self.writer.add_histogram(module_name + '/weights/' +name, param.data, global_step = current_epoch)

    def infer(self, warp, transform):    
        #output preview
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')

        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_transform = transform[0]
        self.last_transform_tensor = torch.unsqueeze(reshaped_t[0], 0)
        
        with torch.no_grad():
            
            t = [0]
            
            t[0] = torch.index_select(reshaped_t, 1, torch.tensor([6, 7])).to(self.gpu_device)
            
            pred = None
            loss = 0.0
            for i in range(self.model_length):
                self.model[i].eval()
                pred = self.model[i](warp_gpu)
                loss = loss + self.singular_loss(pred, t[i])
            
            pred = np.ndarray.flatten(pred[0].cpu().numpy())
            return pred, loss.cpu().data
    
    def get_batch_loss(self):
        return self.batch_loss
    
    def get_state_dicts(self, index):
        return self.model[index].state_dict(), self.optimizers[index].state_dict()
    
    def load_saved_states(self, index, model_dict, optimizer_dict):
        self.model[index].load_state_dict(model_dict)
        self.optimizers[index].load_state_dict(optimizer_dict)
    
    def get_last_warp_img(self):
        return self.last_warp_img
    
    def get_last_warp_tensor(self):
        return self.last_warp_tensor

    def get_last_transform(self):
        return self.last_transform
    
    def get_last_transform_tensor(self):
        return self.last_transform_tensor
    
    def get_name(self):
        return self.name
        