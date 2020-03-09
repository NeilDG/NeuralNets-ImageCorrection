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

class ConcatTrainer:
    
    def __init__(self, name, gpu_device, writer, lr = 0.05, weight_decay = 0.0):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.writer = writer
        
        self.model = [0, 0, 0, 0, 0, 0]
        self.optimizers = [0, 0, 0, 0, 0, 0]
        for i in range(6):
            self.model[i] = concat_cnn.ConcatCNN()
            self.model[i].to(self.gpu_device)
            self.optimizers[i] = optim.Adam(self.model[i].parameters(), lr = self.lr, weight_decay = weight_decay)
        
        self.loss_weights = [1.0, 1.15, 2.5]
    
    def custom_loss(self, pred_group, target_group, weight_group):
        mse_loss = torch.nn.MSELoss(reduction = 'sum')
        
        loss_group = [0.0, 0.0, 0.0]
        loss_group[0] = mse_loss(pred_group[0], target_group[0]) * weight_group[0]
        loss_group[1] = mse_loss(pred_group[1], target_group[1]) * weight_group[1]
        loss_group[2] = mse_loss(pred_group[2], target_group[2]) * weight_group[2]
        return loss_group[0] + loss_group[1] + loss_group[2]
    
    def singular_loss(self, pred, target):
        mse_loss = torch.nn.MSELoss(reduction = 'sum')
        return mse_loss(pred, target)
    
    def train(self, warp, warp_orig, transform):
        self.model[0].train(); self.model[1].train(); self.model[2].train();
        
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        warp_img_orig = warp_orig[0,:,:,:].numpy()
        warp_img_orig = np.moveaxis(warp_img_orig, -1, 0)
        warp_img_orig = np.moveaxis(warp_img_orig, -1, 0) #for properly displaying image in matplotlib
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
        t = [0, 0, 0, 0, 0, 0]
        t[0] = torch.index_select(reshaped_t, 1, torch.tensor([0])).to(self.gpu_device)
        t[1] = torch.index_select(reshaped_t, 1, torch.tensor([1])).to(self.gpu_device)
        t[2] = torch.index_select(reshaped_t, 1, torch.tensor([3])).to(self.gpu_device)
        t[3] = torch.index_select(reshaped_t, 1, torch.tensor([4])).to(self.gpu_device)
        t[4] = torch.index_select(reshaped_t, 1, torch.tensor([6])).to(self.gpu_device)
        t[5] = torch.index_select(reshaped_t, 1, torch.tensor([7])).to(self.gpu_device)
        
        #0 1 2 3 4 5
        self.batch_loss = 0.0
        for i in range(6):
            pred = self.model[i](warp_gpu)
            self.optimizers[i].zero_grad()
            loss = self.singular_loss(pred, t[i])
            loss.backward()
            self.optimizers[i].step();
            self.batch_loss = self.batch_loss + loss.cpu().data     
        
        self.last_warp_img = warp_img
        self.last_warp_img_orig = warp_img_orig
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_warp_tensor_orig = torch.unsqueeze(warp_orig[0,:,:,:], 0)
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

    def infer(self, warp, warp_orig, transform):    
        #output preview
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        warp_img_orig = warp_orig[0,:,:,:].numpy()
        warp_img_orig = np.moveaxis(warp_img_orig, -1, 0)
        warp_img_orig = np.moveaxis(warp_img_orig, -1, 0) #for properly displaying image in matplotlib
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')

        self.last_warp_img = warp_img
        self.last_warp_img_orig = warp_img_orig
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_warp_tensor_orig = torch.unsqueeze(warp_orig[0,:,:,:], 0)
        self.last_transform = transform[0]
        self.last_transform_tensor = torch.unsqueeze(reshaped_t[0], 0)
        
        self.model[0].eval(); self.model[1].eval(); self.model[2].eval();
        with torch.no_grad():
            
            t = [0, 0, 0, 0, 0, 0]
            t[0] = torch.index_select(reshaped_t, 1, torch.tensor([0])).to(self.gpu_device)
            t[1] = torch.index_select(reshaped_t, 1, torch.tensor([1])).to(self.gpu_device)
            t[2] = torch.index_select(reshaped_t, 1, torch.tensor([3])).to(self.gpu_device)
            t[3] = torch.index_select(reshaped_t, 1, torch.tensor([4])).to(self.gpu_device)
            t[4] = torch.index_select(reshaped_t, 1, torch.tensor([6])).to(self.gpu_device)
            t[5] = torch.index_select(reshaped_t, 1, torch.tensor([7])).to(self.gpu_device)
            
            pred = [0,0,0,0,0,0]
            loss = 0.0
            for i in range(6):
                pred[i] = self.model[i](warp_gpu)
                loss = loss + self.singular_loss(pred[i], t[i])
                
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
    
    def get_last_warp_tensor_orig(self):
        return self.last_warp_tensor_orig

    def get_last_transform(self):
        return self.last_transform
    
    def get_last_transform_tensor(self):
        return self.last_transform_tensor
    
    def get_last_warp_img_orig(self):
        return self.last_warp_img_orig
    
    def get_name(self):
        return self.name
        