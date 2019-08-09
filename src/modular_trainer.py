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
    
    def __init__(self, name, gpu_device, batch_size, writer, gt_index = 0, lr = 0.05):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.batch_size = batch_size
        self.writer = writer
        self.gt_index = gt_index
        
        self.model = warp_cnn.WarpCNN()
        self.model.to(self.gpu_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 0.0)
        self.loss_func = torch.nn.MSELoss(reduction = 'sum')
       
    #Specify the gt_index in the vector M to used as ground-truth
    #Trains for 1 epoch
    def train(self, gt_index, current_epoch, warp, transform):
        self.model.train()
        self.gt_index = gt_index
        warp_gpu = warp.to(self.gpu_device)
        warp_img = warp[0,:,:,:].numpy()
        warp_img = np.moveaxis(warp_img, -1, 0)
        warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
        revised_t = torch.reshape(reshaped_t[:,gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
        
        self.optimizer.zero_grad()
        pred = self.model(warp_gpu)
        
        loss = self.loss_func(pred, revised_t)
        loss.backward()
        self.optimizer.step()
        
        #print("[", self.name, "] Training loss: ", loss)
        self.last_warp_img = warp_img
        self.last_warp_tensor = torch.unsqueeze(warp[0,:,:,:], 0)
        self.last_transform = transform[0]
        self.last_transform_tensor = torch.unsqueeze(revised_t[0], 0)
        self.batch_loss = loss.cpu().data
    
    def log_weights(self, current_epoch):
        #log update in weights
        self.writer.add_histogram(self.name + '/weights/fc', self.model.fc.weight.data, global_step = current_epoch)
#        self.writer.add_histogram(self.name + '/weights/conv7', self.model.conv7.weight.data, global_step = current_epoch)
#        self.writer.add_histogram(self.name + '/weights/conv6', self.model.conv6.weight.data, global_step = current_epoch)
#        self.writer.add_histogram(self.name + '/weights/conv5', self.model.conv5.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv4', self.model.conv4.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv3', self.model.conv3.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv2', self.model.conv2.weight.data, global_step = current_epoch)
        self.writer.add_histogram(self.name + '/weights/conv1', self.model.conv1.weight.data, global_step = current_epoch)
    
    def single_infer(self, warp_tensor, ground_truth_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_tensor.to(self.gpu_device))
            loss = self.loss_func(pred, ground_truth_tensor)
            return pred[0].cpu().numpy()[0], loss.cpu().data #return 1 sample of prediction
    
    def blind_infer(self, warp_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_tensor.to(self.gpu_device))
            return pred[0].cpu().numpy()[0] #return 1 sample of prediction
    
    def batch_infer(self, warp_tensor, ground_truth_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            reshaped_t = torch.reshape(ground_truth_tensor, (np.size(ground_truth_tensor, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = torch.reshape(reshaped_t[:,self.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
        
            pred = self.model(warp_tensor.to(self.gpu_device))
            loss = self.loss_func(pred, revised_t)
            
            #replace last attribute tensors
            warp_img = warp_tensor[0,:,:,:].numpy()
            warp_img = np.moveaxis(warp_img, -1, 0)
            warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
            self.last_warp_img = warp_img
            self.last_warp_tensor = torch.unsqueeze(warp_tensor[0,:,:,:], 0)
            self.last_transform = ground_truth_tensor[0]
            self.last_transform_tensor = torch.unsqueeze(revised_t[0], 0)
            return pred[0].cpu().numpy(), loss.cpu().data #return 1 sample of prediction
    
    
    def get_last_warp_img(self):
        return self.last_warp_img
    
    def get_last_warp_tensor(self):
        return self.last_warp_tensor

    def get_last_transform(self):
        return self.last_transform
    
    def get_last_transform_tensor(self):
        return self.last_transform_tensor
    
    def get_batch_loss(self):
        return self.batch_loss
    
    def get_state_dicts(self):
        return self.model.state_dict(), self.optimizer.state_dict()
    
    def load_saved_states(self, model_dict, optimizer_dict):
        self.model.load_state_dict(model_dict)
        self.optimizer.load_state_dict(optimizer_dict)
    
    def get_name(self):
        return self.name
    
    def get_model_layer(self, index):
        return self.model.get_layer_activation(index = index)
    
    def flag_visualize_layer(self, flag):
        self.model.flag_visualize_layer(flag)
        