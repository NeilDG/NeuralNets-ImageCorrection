# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:38:10 2019

Main starting point for warp image training
@author: delgallegon
"""

from model import warp_cnn
from model import torch_image_loader as loader
from utils import generate_misaligned as gm
import torch
from torch import optim
import numpy as np
import cv2
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

LR = 0.001
num_epoch = 500
BATCH_SIZE = 8


def show_transform_image(rgb, M, ground_truth_M):
    M = np.append(M, [1.0])
    M = np.reshape(M, (3,3))
    #print("M predicted contents: ", M, "Ground truth: ", ground_truth_M)
    print("Ground truth")
    #result = cv2.perspectiveTransform(rgb, ground_truth_M.numpy())
    result = cv2.warpPerspective(rgb, ground_truth_M.numpy(), (np.shape(rgb)[1], np.shape(rgb)[0]))
    plt.imshow(result)
    plt.show()
    
    print("Predicted warp")
    #result = cv2.perspectiveTransform(rgb, ground_truth_M.numpy())
    result = cv2.warpPerspective(rgb, M, (np.shape(rgb)[1], np.shape(rgb)[0]))
    plt.imshow(result)
    plt.show()

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
    
    #load checkpoints
    CHECKPATH = 'tmp/warp_cnn_0620.pt'
    if(False):
        checkpoint = torch.load(CHECKPATH)
        cnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    
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
            revised_t = revised_t[:, 0:8].to(gpu_dev)
            #print("Revised T type: ", revised_t.type())
            
            optimizer.zero_grad() #reset gradient computer
            pred = cnn(warp_gpu)
            pred = pred * 50.0 #amplify difference
            revised_t = revised_t * 50.0
            
            loss = loss_func(pred, revised_t)
            loss.backward()
            optimizer.step()
            accum_loss = accum_loss + loss.cpu().data
            
            if(batch_idx % 25 == 0):
                print("Batch id: ", batch_idx, "Loss: ", loss)
                #writer.add_scalar('warp_exp/Batch_MSE_Loss_' +str(epoch + 1), loss, global_step = (batch_idx + 1))
                #writer.close()
        
        train_ave_loss = accum_loss / (batch_idx + 1.0)
        
        writer.add_histogram('warp_exp_2/weights/weights_fc', cnn.fc.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv9', cnn.conv6.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv8', cnn.conv6.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv7', cnn.conv6.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv6', cnn.conv6.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv5', cnn.conv5.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv4', cnn.conv4.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv3', cnn.conv3.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv2', cnn.conv2.weight.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/weights/weights_conv1', cnn.conv1.weight.data, global_step = (epoch + 1))
        
        writer.add_histogram('warp_exp_2/bias/bias_fc', cnn.fc.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv9', cnn.conv6.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv8', cnn.conv6.bias.data, global_step = (epoch + 1))
        writer.add_histogram('warp_exp_2/bias/bias_conv7', cnn.conv6.bias.data, global_step = (epoch + 1))
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
            predict_M_list = []
            for batch_idx, (rgb, warp, transform) in enumerate(loader.load_test_dataset(batch_size = BATCH_SIZE)):
                warp_img = warp[0,:,:,:].numpy()
                warp_img = np.moveaxis(warp_img, -1, 0)
                warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
                
                print("Showing image for batch ID: ", batch_idx)
                
                
                pred = cnn(warp.to(gpu_dev))
                predict_M_list.append(pred[0].cpu().numpy())
                
                revised_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
                revised_t = revised_t[:, 0:8].to(gpu_dev)
            
                #note validation loss
                pred = pred * 50.0
                revised_t = revised_t * 50.0
                loss = loss_func(pred, revised_t)
                accum_loss = accum_loss + loss.cpu().data
                
                #if((epoch + 1) % 20 != 0): #only save a batch every 25 epochs
                    #break
            
            plt.imshow(warp_img)
            plt.show()
            show_transform_image(warp_img, M = pred[0].cpu().numpy(), ground_truth_M = transform[0])
            validate_ave_loss = accum_loss / (batch_idx + 1.0)
            writer.add_scalars('warp_exp_2/MSE_loss', {'training_loss' :train_ave_loss, 'validation_loss' : validate_ave_loss}, global_step = (epoch + 1)) #plot validation loss
            writer.close()
            
            print("Current epoch: ", (epoch + 1), " Training loss: ", train_ave_loss, "Validation loss: ", validate_ave_loss)
            if((epoch + 1) % 10 == 0): #only save a batch every 25 epochs
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
