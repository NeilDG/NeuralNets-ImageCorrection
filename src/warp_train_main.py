# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:38:10 2019

Main starting point for warp image training
@author: delgallegon
"""

from model import warp_cnn
from model import torch_image_loader as loader
from utils import generate_misaligned as gm
import global_vars as gv
import torch
import math
from torch import optim
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

LR = 0.001
num_epoch = 500
BATCH_SIZE = 8


def show_transform_image(rgb, M, ground_truth_M):
    #M = M / gv.WARPING_CONSTANT
    #ground_truth_M = ground_truth_M / gv.WARPING_CONSTANT
    
    pred_M = np.copy(ground_truth_M)
    pred_M[0,1] = M
    
    #hardcode muna
    #M = np.append(M, [1.0])
    #M = np.reshape(M, (3,3))
    #result = cv2.perspectiveTransform(rgb, ground_truth_M.numpy())
    result = cv2.warpPerspective(rgb, ground_truth_M.numpy(), (np.shape(rgb)[1], np.shape(rgb)[0]))
    plt.title("Ground truth")
    plt.imshow(result)
    plt.show()
    
    #result = cv2.perspectiveTransform(rgb, ground_truth_M.numpy())
    result = cv2.warpPerspective(rgb, pred_M, (np.shape(rgb)[1], np.shape(rgb)[0]))
    plt.title("Predicted warp")
    plt.imshow(result)
    plt.show()
    
    print("M predicted contents: ", pred_M, "Ground truth: ", ground_truth_M, "Norm: ", np.linalg.norm((M - ground_truth_M.numpy())))

def normalize(tensor_v, reference_tensor):
    min_v = torch.min(reference_tensor * 500)
    range_v = torch.max(reference_tensor * 500) - min_v
    if range_v > 0:
        normalised = (tensor_v - min_v) / range_v
    else:
        normalised = torch.zeros(tensor_v.size())
    
    return normalised

def start_train(gpu_dev):
    
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')
    
    #load model
    cnn = warp_cnn.WarpCNN()
    cnn.to(gpu_dev)
    optimizer = optim.Adam(cnn.parameters(),lr = LR)
    loss_func = torch.nn.MSELoss(reduction = 'sum')
    pairwise_dist = torch.nn.PairwiseDistance()
    
    #load checkpoints
    CHECKPATH = 'tmp/warp_cnn_0621.pt'
    if(False):
        checkpoint = torch.load(CHECKPATH)
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Loaded checkpt ",CHECKPATH, "Current epoch: ", epoch)
    
    for epoch in range(num_epoch):
        cnn.train()
        
        accum_loss = 0.0
        train_ave_loss = 0.0
        print("Started training per batch.")
        print("Conv1 biases: ", cnn.conv1.bias.data)
        print("FC biases: ", cnn.fc.bias.data)
        for batch_idx, (rgb, warp, transform) in enumerate(loader.load_dataset(batch_size = BATCH_SIZE)):
            warp_gpu = warp.to(gpu_dev)
            warp_img = warp[0,:,:,:].numpy()
            warp_img = np.moveaxis(warp_img, -1, 0)
            warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
            
            rgb_img = rgb[0,:,:,:].numpy()
            rgb_img = np.moveaxis(rgb_img, -1, 0)
            rgb_img = np.moveaxis(rgb_img, -1, 0) #for properly displaying image in matplotlib
            
            revised_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = revised_t[:,1].to(gpu_dev)
            #print("Revised T type: ", revised_t.type())
            
            optimizer.zero_grad() #reset gradient computer
            pred = cnn(warp_gpu)
            
            loss = loss_func(pred, revised_t)
            loss.backward()
            optimizer.step()
            accum_loss = accum_loss + loss.cpu().data
            
            if(batch_idx % 25 == 0):
                print("Batch id: ", batch_idx, "Loss: ", loss)
        
        train_ave_loss = accum_loss / (batch_idx + 1.0)
        
        writer.add_histogram('warp_exp_2/weights/weights_fc', cnn.fc.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv9', cnn.conv9.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv8', cnn.conv8.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv7', cnn.conv7.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv6', cnn.conv6.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv5', cnn.conv5.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv4', cnn.conv4.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv3', cnn.conv3.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv2', cnn.conv2.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv1', cnn.conv1.weight.data, global_step = (epoch + 1))
        
        writer.add_histogram('warp_exp_2/bias/bias_fc', cnn.fc.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv9', cnn.conv9.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv8', cnn.conv8.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv7', cnn.conv7.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv6', cnn.conv6.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv5', cnn.conv5.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv4', cnn.conv4.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv3', cnn.conv3.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv2', cnn.conv2.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv1', cnn.conv1.bias.data, global_step = (epoch + 1))
        
        #evaluate predictions
        accum_loss = 0.0
        validate_ave_loss = 0.0
        cnn.eval()
        with torch.no_grad():
            pred = cnn(warp.to(gpu_dev))
            print("Training set preview")
            plt.title("Input image")
            plt.imshow(warp_img)
            plt.show()
            show_transform_image(warp_img, M = pred[0].cpu().numpy(), ground_truth_M = transform[0])
            
            predict_M_list = []
            for batch_idx, (rgb, warp, transform) in enumerate(loader.load_test_dataset(batch_size = BATCH_SIZE)):
                warp_img = warp[0,:,:,:].numpy()
                warp_img = np.moveaxis(warp_img, -1, 0)
                warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
                
                #print("Showing image for batch ID: ", batch_idx)
                
                pred = cnn(warp.to(gpu_dev))
                predict_M_list.append(pred[0].cpu().numpy())
                
                revised_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
                revised_t = revised_t[:, 1].to(gpu_dev)
            
                #note validation loss
                loss = loss_func(pred, revised_t)
                accum_loss = accum_loss + loss.cpu().data
                
                #if((epoch + 1) % 20 != 0): #only save a batch every 25 epochs
                    #break
            
            print("Validation set preview")
            plt.title("Input image")
            plt.imshow(warp_img)
            plt.show()
            show_transform_image(warp_img, M = pred[0].cpu().numpy(), ground_truth_M = transform[0])
            validate_ave_loss = accum_loss / (batch_idx + 1.0)
            writer.add_scalars('warp_exp_2/MSE_loss', {'training_loss' :train_ave_loss, 'validation_loss' : validate_ave_loss}, global_step = (epoch + 1)) #plot validation loss
            writer.close()
            
            print("Current epoch: ", (epoch + 1), " Training loss: ", train_ave_loss, "Validation loss: ", validate_ave_loss)
            if((epoch + 1) % 10 == 0): #only save a batch every X epochs
                gm.save_predicted_transforms(predict_M_list, 0) #use epoch value if want to save per epoch
                torch.save(cnn.state_dict(), CHECKPATH)
                torch.save({'epoch': epoch,
                            'model_state_dict': cnn.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, CHECKPATH)
      
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_train(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
