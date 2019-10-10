# -*- coding: utf-8 -*-
"""
A trainer for the autoencoder
Created on Thu Oct 10 20:40:53 2019

@author: delgallegon
"""

from model import warp_autoencoder
from loaders import torch_image_loader as loader
from utils import generate_misaligned as gm
import global_vars as gv
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

class AutoEncoderTrainer:
    
    def __init__(self, name, gpu_device, batch_size, writer, gt_index = 0, lr = 0.05):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.batch_size = batch_size
        self.writer = writer
        self.gt_index = gt_index
        
        self.model = warp_autoencoder.WarpAutoEncoder()
        self.model.to(self.gpu_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 0.0)
        self.loss_func = torch.nn.MSELoss(reduction = 'sum')
        
    
    #Specify the gt_index in the vector M to used as ground-truth
    #Trains for 1 epoch
    def train(self, input_img, ground_truth_img, current_epoch):
        self.model.train()
        warp_gpu = input_img.to(self.gpu_device)
        gt_gpu = ground_truth_img.to(self.gpu_device)
        warp_img = input_img[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        gt_img = ground_truth_img[0,:,:,:].numpy()
        gt_img = np.moveaxis(gt_img, -1, 0)
        gt_img = np.moveaxis(gt_img, -1, 0) #for properly displaying image in matplotlib
        
        self.optimizer.zero_grad()
        pred = self.model(warp_gpu)
        
        loss = self.loss_func(pred, gt_gpu)
        loss.backward()
        self.optimizer.step()
        
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(input_img[0,:,:,:], 0)
        self.ground_truth_img = gt_img
        self.ground_truth_tensor = torch.unsqueeze(ground_truth_img[0,:,:,:], 0)
        self.batch_loss = loss.cpu().data
    
    #return 1 sample of prediction. Used for visualization purposes
    def single_infer(self, warp_tensor, ground_truth_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_tensor.to(self.gpu_device))
            loss = self.loss_func(pred, ground_truth_tensor.to(self.gpu_device))
            self.ground_truth_img = ground_truth_tensor[:,:,:].numpy()
            return pred.cpu().numpy()[0], loss.cpu().data 
    
    #return 1 batch of prediction
    def batch_infer(self, warp_tensor, ground_truth_tensor):    
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_tensor.to(self.gpu_device))
            loss = self.loss_func(pred, ground_truth_tensor.to(self.gpu_device))
            self.ground_truth_img = ground_truth_tensor[0,:,:,:].numpy()
            self.ground_truth_img = np.moveaxis(self.ground_truth_img, -1, 0)
            self.ground_truth_img = np.moveaxis(self.ground_truth_img, -1, 0) #for properly displaying image in matplotlib
            return pred.cpu().numpy()[0], loss.cpu().data 
    
    def log_weights(self, current_epoch):
        #log update in weights
        self.writer.add_histogram(self.name + '/weights/conv1', self.model.conv1.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv1_T', self.model.conv1_T.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv2', self.model.conv2.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv2_T', self.model.conv2_T.weight.data, global_step = current_epoch)
    
    def get_last_warp_img(self):
        return self.last_warp_img
    
    def get_last_ground_truth_img(self):
        return self.ground_truth_img
    
    def get_last_warp_tensor(self):
        return self.last_warp_tensor
    
    def get_last_ground_truth_tensor(self):
        return self.ground_truth_tensor
    
    def get_batch_loss(self):
        return self.batch_loss
    
    def get_state_dicts(self):
        return self.model.state_dict(), self.optimizer.state_dict()
    
    def load_saved_states(self, model_dict, optimizer_dict):
        self.model.load_state_dict(model_dict)
        self.optimizer.load_state_dict(optimizer_dict)
    
    def get_name(self):
        return self.name