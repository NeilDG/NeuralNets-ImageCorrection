# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:28:58 2019

Class for producing intermediate results
@author: delgallegon
"""

from model import warp_cnn
import torch
import numpy as np

class IntermediateTrainer:
    
    def __init__(self, name, gpu_device, writer):
        self.gpu_device = gpu_device
        self.name = name
        self.writer = writer
        
        self.model = warp_cnn.WarpCNN()
        self.model.to(self.gpu_device)
    
    def produce_intermediate(self, gt_index, warp, transform):
        self.model.train()
        self.gt_index = gt_index
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
        revised_t = torch.reshape(reshaped_t[:,gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
        
        pred = self.model(warp_gpu)
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_transform = transform[0]
        self.last_transform_tensor = torch.unsqueeze(revised_t[0], 0)
        
        return pred
    
    def single_infer(self, warp_tensor, ground_truth_tensor):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_tensor.to(self.gpu_device))
            
            reshaped_t = torch.reshape(ground_truth_tensor, (np.size(ground_truth_tensor, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = torch.reshape(reshaped_t[:,self.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
                
            warp_img = warp_tensor[0,:,:,:].numpy()
            warp_img = np.moveaxis(warp_img, -1, 0)
            warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
            self.last_warp_img = warp_img
            self.last_warp_tensor = torch.unsqueeze(warp_tensor[0,:,:,:], 0)
            self.last_transform = ground_truth_tensor[0]
            self.last_transform_tensor = torch.unsqueeze(revised_t[0], 0)
            
            return pred
    
    def batch_infer(self, warp_tensor, ground_truth_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            reshaped_t = torch.reshape(ground_truth_tensor, (np.size(ground_truth_tensor, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = torch.reshape(reshaped_t[:,self.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
        
            pred = self.model(warp_tensor.to(self.gpu_device))
            
            #replace last attribute tensors
            warp_img = warp_tensor[0,:,:,:].numpy()
            warp_img = np.moveaxis(warp_img, -1, 0)
            warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
            self.last_warp_img = warp_img
            self.last_warp_tensor = torch.unsqueeze(warp_tensor[0,:,:,:], 0)
            self.last_transform = ground_truth_tensor[0]
            self.last_transform_tensor = torch.unsqueeze(revised_t[0], 0)
            
            return pred
    
    def log_weights(self, current_epoch):
        #log update in weights
        self.writer.add_histogram(self.name + '/weights/fc', self.model.fc.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv4', self.model.conv4.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv3', self.model.conv3.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv2', self.model.conv2.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv1', self.model.conv1.weight.data, global_step = current_epoch)
        
    def get_gt_index(self):
        return self.gt_index
    
    def get_last_warp_img(self):
        return self.last_warp_img
    
    def get_last_warp_tensor(self):
        return self.last_warp_tensor

    def get_last_transform(self):
        return self.last_transform
    
    def get_last_transform_tensor(self):
        return self.last_transform_tensor
    
    def get_state_dicts(self):
        return self.model.state_dict()
    
    def load_saved_states(self, model_dict):
        self.model.load_state_dict(model_dict)
    
    def get_name(self):
        return self.name