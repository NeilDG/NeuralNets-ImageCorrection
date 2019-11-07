# -*- coding: utf-8 -*-
"""
Class for modular training
Created on Wed Jul 17 11:10:28 2019

@author: delgallegon
"""
from model import warp_cnn_joiner
from loaders import torch_image_loader as loader
from utils import generate_misaligned as gm
import global_vars as gv
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

class ModularTrainer:
    
    def __init__(self, name, gpu_device, writer, gt_index = 0, lr = 0.05):
        self.gpu_device = gpu_device
        self.lr = lr
        self.name = name
        self.writer = writer
        self.gt_index = gt_index
        
        self.model = warp_cnn_joiner.JoinerCNN()
        self.model.to(self.gpu_device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 0.0)
        self.loss_func = torch.nn.MSELoss(reduction = 'sum')
       
    #Specify the gt_index in the vector M to used as ground-truth
    #Trains for 1 epoch
    def train(self, gt_index, intermediate_tensor, transform):
        self.model.train()
        self.gt_index = gt_index
       
        pred = self.model(intermediate_tensor)
        
        reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
        revised_t = torch.reshape(reshaped_t[:,gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
        
        self.optimizer.zero_grad()
        loss = self.loss_func(pred, revised_t)
        loss.backward()
        self.optimizer.step()
        self.batch_loss = loss.cpu().data
    
    def log_weights(self, current_epoch):
        #log update in weights
        self.writer.add_histogram(self.name + '/weights/concat1', self.model.concat1.weight.data, global_step = current_epoch) 

    def single_infer(self, intermediate_tensor, ground_truth_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            pred = self.model(intermediate_tensor)
            #print("Pred shape: ", np.shape(pred), "Gt shape: ", np.shape(ground_truth_tensor))
            loss = self.loss_func(pred, ground_truth_tensor)
            return pred[0].cpu().numpy()[0], loss.cpu().data #return 1 sample of prediction
    
    def blind_infer(self, warp_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            pred = self.model(warp_tensor.to(self.gpu_device))
            return pred[0].cpu().numpy()[0] #return 1 sample of prediction
    
    def batch_infer(self, intermediate_tensor, ground_truth_tensor):    
        #output preview
        self.model.eval()
        with torch.no_grad():
            reshaped_t = torch.reshape(ground_truth_tensor, (np.size(ground_truth_tensor, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = torch.reshape(reshaped_t[:,self.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(self.gpu_device)
        
            pred = self.model(intermediate_tensor)
            loss = self.loss_func(pred, revised_t)
            
            return pred[0].cpu().numpy(), loss.cpu().data #return 1 sample of prediction
    
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
        