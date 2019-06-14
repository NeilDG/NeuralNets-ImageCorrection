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
from torch import optim
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision import transforms

LR = 0.05
num_epoch = 100
BATCH_SIZE = 8

def load_dataset():
    rgb_list, warp_list, transform_list = loader.assemble_train_data()
    print("Length of train images: ", len(rgb_list), len(warp_list))
    
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    train_dataset = image_dataset.TorchImageDataset(rgb_list, warp_list, transform_list, image_transform_op = generic_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True
    )
    
    return train_loader

def load_test_dataset():
    rgb_list, warp_list, transform_list = loader.assemble_test_data()
    print("Length of test images: ", len(rgb_list), len(warp_list))
    
    generic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    test_dataset = image_dataset.TorchImageDataset(rgb_list, warp_list, transform_list, image_transform_op = generic_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True
    )
    
    return test_loader

def show_transform_image(rgb, M):
    M = np.append(M, [1.0])
    M = np.reshape(M, (3,3))
    print("M predicted contents: ", M)
    result = cv2.warpPerspective(rgb, M, (np.shape(rgb)[1], np.shape(rgb)[0]))
    plt.imshow(result)
    plt.show()

def start_train(gpu_dev):
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')
    
    #load model
    cnn = warp_cnn.WarpCNN()
    cnn.to(gpu_dev)
    optimizer = optim.Adam(cnn.parameters(),lr = LR)
    loss_func = torch.nn.MSELoss()
    
    for epoch in range(num_epoch):
        cnn.train()
        
        accum_loss = 0.0
        ave_loss = 0.0
        print("Started training per batch.")
        for batch_idx, (rgb, warp, transform) in enumerate(load_dataset()):
            warp_gpu = warp.to(gpu_dev)
            warp_img = warp[0,:,:,:].numpy()
            warp_img = np.moveaxis(warp_img, -1, 0)
            warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
            
            revised_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = revised_t[:, 0:8].to(gpu_dev)
            #print("Revised T type: ", revised_t.type())
            
            optimizer.zero_grad() #reset gradient computer
            pred = cnn(warp_gpu)
            loss = loss_func(pred, revised_t)
            loss.backward()
            optimizer.step()
            accum_loss = accum_loss + loss.cpu().data
            
            if(batch_idx % 25 == 0):
                print("Batch id: ", batch_idx, "Loss: ", loss)
                writer.add_scalar('warp_exp/Batch_MSE_Loss', loss, global_step = (batch_idx + 1))
                writer.close()
        
        ave_loss = accum_loss / (batch_idx + 1.0)
        plt.plot((epoch + 1), ave_loss, "-o")
        plt.show()
        
        
        print("Current epoch: ", (epoch + 1), " Loss: ", ave_loss)
        writer.add_scalar('warp_exp/MSE_loss', ave_loss, global_step = (epoch + 1))
        writer.close()
        
        #evaluate predictions
        cnn.eval()
        with torch.no_grad():
            for batch_idx, (rgb, warp, transform) in enumerate(load_test_dataset()):
                rgb_img = warp[0,:,:,:].numpy()
                rgb_img = np.moveaxis(rgb_img, -1, 0)
                rgb_img = np.moveaxis(rgb_img, -1, 0) #for properly displaying image in matplotlib
                
                print("Showing image for batch ID: ", batch_idx)
                plt.imshow(rgb_img)
                plt.show()
                
                batch_pred = cnn(rgb.to(gpu_dev))
                show_transform_image(rgb_img, batch_pred[0].cpu().numpy())
      
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_train(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
