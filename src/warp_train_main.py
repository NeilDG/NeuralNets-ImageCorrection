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
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

def load_dataset():
    rgb_list, warp_list, transform_list = loader.assemble_train_data()
    
    print("Length of images: ", len(rgb_list), len(warp_list))
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_dataset = image_dataset.TorchImageDataset(rgb_list, warp_list, transform_list, image_transform_op = generic_transform)
    
    train_dataset.__getitem__(0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True
    )
    return train_loader

def start_train():
    for batch_idx, (rgb, warp, transform) in enumerate(load_dataset()):
        print("Batch id:", batch_idx, "RGB type: ", rgb[0,:,:,:].dtype, "Shape:" ,np.shape(rgb[0,:,:,:].numpy()))
        rgb_img = rgb[0,:,:,:].numpy()
        rgb_img = np.moveaxis(rgb_img, -1, 0)
        rgb_img = np.moveaxis(rgb_img, -1, 0)
        print("New shape: ", np.shape(rgb_img))
        #print("Values: ", rgb[0,:,:,:].numpy())
        plt.imshow(rgb_img)
        plt.show()
        
def main():
    
    start_train()
    cnn = warp_cnn.WarpCNN()
    print(cnn)

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
