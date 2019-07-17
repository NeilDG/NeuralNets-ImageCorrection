# -*- coding: utf-8 -*-
"""
Class for modular training
Created on Wed Jul 17 11:10:28 2019

@author: delgallegon
"""
from model import warp_cnn
from loaders import torch_image_loader as loader
from utils import generate_misaligned as gm
import global_vars as gv
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

class ModularTrainer:
    
    def __init__(self, name, gpu_device, batch_size, writer, lr = 0.05):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.batch_size = batch_size
        self.writer = writer
        
        self.model = warp_cnn.WarpCNN()
        self.model.to(self.gpu_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.loss_func = torch.nn.MSELoss(reduction = 'sum')
       
    #Specify the gt_index in the vector M to used as ground-truth
    #Trains for 1 epoch
    def train(self, gt_index, current_epoch):
        self.model.train()
        accum_loss = 0.0
        self.train_ave_loss = 0.0
        print("[", self.name, "] Started training per batch")
        
        for batch_idx, (rgb, warp, transform) in enumerate(loader.load_dataset(batch_size = self.batch_size)):
            warp_gpu = warp.to(self.gpu_device)
            warp_img = warp[0,:,:,:].numpy()
            warp_img = np.moveaxis(warp_img, -1, 0)
            warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
            
            rgb_img = rgb[0,:,:,:].numpy()
            rgb_img = np.moveaxis(rgb_img, -1, 0)
            rgb_img = np.moveaxis(rgb_img, -1, 0) #for properly displaying image in matplotlib
            
            reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = torch.reshape(reshaped_t[:,gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
            
            self.optimizer.zero_grad()
            pred = self.model(warp_gpu)
            
            loss = self.loss_func(pred, revised_t)
            loss.backward()
            self.optimizer.step()
            accum_loss = accum_loss + loss.cpu().data
            
            if(batch_idx % 25 == 0):
                print("[", self.name, "] Batch id: ", batch_idx, "Loss: ", loss)
                self.train_ave_loss = accum_loss / (batch_idx + 1.0)
            
        #log update in weights
        self.writer.add_histogram(self.name + '/weights/fc', self.model.fc.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv5', self.model.conv5.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv4', self.model.conv4.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv3', self.model.conv3.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv2', self.model.conv2.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv1', self.model.conv1.weight.data, global_step = current_epoch)
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_transform = transform[0]
    
    def infer(self, warp_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_tensor.to(self.gpu_device))
            return pred[0].cpu().numpy() #return 1 sample of prediction
    
    def get_last_warp_img(self):
        return self.last_warp_img
    
    def get_last_warp_tensor(self):
        return self.last_warp_tensor

    def get_last_transform(self):
        return self.last_transform
    
    def get_average_train_loss(self):
        return self.train_ave_loss
    
    def get_name(self):
        return self.name
        