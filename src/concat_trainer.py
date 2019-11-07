# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 19:41:58 2019

@author: delgallegon
"""

from model import concat_cnn
from loaders import torch_image_loader as loader
from utils import generate_misaligned as gm
import global_vars as gv
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

class ConcatTrainer:
    
    def __init__(self, name, gpu_device, writer, lr = 0.05):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.writer = writer
        
        self.model = concat_cnn.ConcatCNN()
        self.model.to(self.gpu_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 0.0)
        self.loss_func = torch.nn.MSELoss(reduction = 'sum')
    
    def train(self, warp, transform):
        self.model.train()
        
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
        revised_t = torch.cat((reshaped_t[:,1:4], reshaped_t[:,5:8]),1).to(self.gpu_device)
        predictions = self.model(warp_gpu)
        
        #print("Prediction size: " ,predictions.size(), " Revised_T size: ", revised_t.size())
        
        self.optimizer.zero_grad()
        loss = self.loss_func(predictions, revised_t)
        loss.backward()
        self.optimizer.step()
        self.batch_loss = loss.cpu().data
        
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_transform = transform[0]
        self.last_transform_tensor = torch.unsqueeze(reshaped_t[0], 0)
    
    def log_weights(self, current_epoch):
        #log update in weights
        weights = list(self.model.parameters())
        print("Weight shape: ", np.shape(weights))
        #self.writer.add_histogram(self.name + '/weights/concat_block1', self.model.concat_block1.weight.data, global_step = current_epoch)

    def infer(self, warp, transform):    
        #output preview
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
        revised_t = torch.cat((reshaped_t[:,1:4], reshaped_t[:,5:8]),1).to(self.gpu_device)
        
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_transform = transform[0]
        self.last_transform_tensor = torch.unsqueeze(reshaped_t[0], 0)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_gpu)
            loss = self.loss_func(pred, revised_t)
            return pred[0].cpu().numpy(), loss.cpu().data #return 1 sample of prediction
    
    def get_batch_loss(self):
        return self.batch_loss
    
    def get_state_dicts(self):
        return self.model.state_dict(), self.optimizer.state_dict()
    
    def load_saved_states(self, model_dict, optimizer_dict):
        self.model.load_state_dict(model_dict)
        self.optimizer.load_state_dict(optimizer_dict)
    
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
        