# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:38:10 2019

Main starting point for warp image training
@author: delgallegon
"""

from model import warp_cnn
from model import torch_image_loader as loader
from model import torch_image_dataset as image_dataset
import torch

def load_dataset():
    rgb_list, warp_list, transform_list = loader.assemble_train_data()
    train_dataset = image_dataset.TorchImageDataset(rgb_list, warp_list, transform_list)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def start_train():
    for batch_idx, (rgb, warp, transform) in enumerate(load_dataset()):
        print(batch_idx, rgb, warp)
        
def main():
    
    start_train()
    cnn = warp_cnn.WarpCNN()
    print(cnn)

main()
