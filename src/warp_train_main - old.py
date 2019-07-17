# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:38:10 2019

Main starting point for warp image training
@author: delgallegon
"""

from model import warp_cnn
from loaders import torch_image_loader as loader
from utils import generate_misaligned as gm
import torch
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

LR = 0.0001
num_epoch = 500
BATCH_SIZE = 40
CNN_VERSION = "cnn_v3.09"


def start_train(gpu_dev):
    
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')
    
    #load model
    cnn = warp_cnn.WarpCNN()
    cnn.to(gpu_dev)
    optimizer = optim.Adam(cnn.parameters(),lr = LR)
    loss_func = torch.nn.MSELoss(reduction = 'sum')
    
    #load second model
    cnn_2 = warp_cnn.WarpCNN();
    cnn_2.to(gpu_dev)
    optimizer_2 = optim.Adam(cnn_2.parameters(),lr = LR)
    
    #third model
    cnn_3 = warp_cnn.WarpCNN();
    cnn_3.to(gpu_dev)
    optimizer_3 = optim.Adam(cnn_3.parameters(),lr = LR)
    
    #fourth model
    cnn_4 = warp_cnn.WarpCNN();
    cnn_4.to(gpu_dev)
    optimizer_4 = optim.Adam(cnn_4.parameters(),lr = LR)
    
    #load checkpoints
    CHECKPATH = 'tmp/' + CNN_VERSION +'.pt'
    if(False):
        checkpoint = torch.load(CHECKPATH)
        cnn.load_state_dict(checkpoint['cnn_1'])
        optimizer.load_state_dict(checkpoint['cnn_1_optimizer_state_dict'])
        cnn_2.load_state_dict(checkpoint['cnn_2'])
        optimizer_2.load_state_dict(checkpoint['cnn_2_optimizer_state_dict'])
        cnn_3.load_state_dict(checkpoint['cnn_3'])
        optimizer_3.load_state_dict(checkpoint['cnn_3_optimizer_state_dict'])
        cnn_4.load_state_dict(checkpoint['cnn_4'])
        optimizer_4.load_state_dict(checkpoint['cnn_4_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Loaded checkpt ",CHECKPATH, "Current epoch: ", epoch)
    
    for epoch in range(num_epoch):
        cnn.train()
        
        accum_loss = 0.0
        train_ave_loss = 0.0
        print("Started training per batch.")
        #print("Conv1 biases: ", cnn.conv1.bias.data)
        #print("FC biases: ", cnn.fc.bias.data)
        for batch_idx, (rgb, warp, transform) in enumerate(loader.load_dataset(batch_size = BATCH_SIZE)):
            warp_gpu = warp.to(gpu_dev)
            warp_img = warp[0,:,:,:].numpy()
            warp_img = np.moveaxis(warp_img, -1, 0)
            warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
            
            rgb_img = rgb[0,:,:,:].numpy()
            rgb_img = np.moveaxis(rgb_img, -1, 0)
            rgb_img = np.moveaxis(rgb_img, -1, 0) #for properly displaying image in matplotlib
            
            reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
            revised_t = torch.reshape(reshaped_t[:,0], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
            
            optimizer.zero_grad() #reset gradient computer
            pred = cnn(warp_gpu)
            
            loss = loss_func(pred, revised_t)
            loss.backward()
            optimizer.step()
            accum_loss = accum_loss + loss.cpu().data
            
            #perform 2nd training
            revised_t = torch.reshape(reshaped_t[:,1], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
            optimizer_2.zero_grad()
            pred_2 = cnn_2(warp_gpu)
            loss_2 = loss_func(pred_2, revised_t)
            loss_2.backward()
            optimizer_2.step()
            accum_loss = accum_loss + loss.cpu().data
            
            #perform 3rd training
            revised_t = torch.reshape(reshaped_t[:,3], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
            optimizer_3.zero_grad()
            pred_3 = cnn_3(warp_gpu)
            loss_3 = loss_func(pred_3, revised_t)
            loss_3.backward()
            optimizer_3.step()
            accum_loss = accum_loss + loss.cpu().data
            
            #perform 3rd training
            revised_t = torch.reshape(reshaped_t[:,4], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
            optimizer_4.zero_grad()
            pred_4 = cnn_4(warp_gpu)
            loss_4 = loss_func(pred_4, revised_t)
            loss_4.backward()
            optimizer_4.step()
            accum_loss = accum_loss + loss.cpu().data
            
            if(batch_idx % 25 == 0):
                print("Batch id: ", batch_idx, 
                      "\n Loss 1: ", loss, 
                      "\n Loss 2: ", loss_2, 
                      "\n Loss 3: ", loss_3, 
                      "\n Loss 4: ", loss_4)
        
        train_ave_loss = accum_loss / (batch_idx + 1.0)
        
        writer.add_histogram(CNN_VERSION +'/weights/weights_fc', cnn.fc.weight.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/weights/weights_conv9', cnn.conv9.weight.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/weights/weights_conv8', cnn.conv8.weight.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/weights/weights_conv7', cnn.conv7.weight.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/weights/weights_conv6', cnn.conv6.weight.data, global_step = (epoch + 1))
#        writer.add_histogram('cnn_v3/weights/weights_conv5', cnn.conv5.weight.data, global_step = (epoch + 1))
#        writer.add_histogram('cnn_v3/weights/weights_conv4', cnn.conv4.weight.data, global_step = (epoch + 1))
        writer.add_histogram(CNN_VERSION +'/weights/weights_conv3', cnn.conv3.weight.data, global_step = (epoch + 1))
        writer.add_histogram(CNN_VERSION +'/weights/weights_conv2', cnn.conv2.weight.data, global_step = (epoch + 1))
        writer.add_histogram(CNN_VERSION +'/weights/weights_conv1', cnn.conv1.weight.data, global_step = (epoch + 1))
        
#        writer.add_histogram('cnn_v3/bias/bias_fc', cnn.fc.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/bias/bias_conv9', cnn.conv9.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/bias/bias_conv8', cnn.conv8.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/bias/bias_conv7', cnn.conv7.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/bias/bias_conv6', cnn.conv6.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/bias/bias_conv5', cnn.conv5.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('warp_exp_2/bias/bias_conv4', cnn.conv4.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('cnn_v3/bias/bias_conv3', cnn.conv3.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('cnn_v3/bias/bias_conv2', cnn.conv2.bias.data, global_step = (epoch + 1))
#        writer.add_histogram('cnn_v3/bias/bias_conv1', cnn.conv1.bias.data, global_step = (epoch + 1))
        
        #evaluate predictions
        accum_loss = 0.0
        validate_ave_loss = 0.0
        cnn.eval(); cnn_2.eval(); cnn_3.eval(); cnn_4.eval()
        with torch.no_grad():
            pred = cnn(warp.to(gpu_dev))
            print("Training set preview")
            plt.title("Input image")
            plt.imshow(warp_img)
            plt.show()
            show_transform_image(warp_img, M1 = pred[0].cpu().numpy(), M2 = pred_2[0].cpu().numpy(), M3 = pred_3[0].cpu().numpy(),
                                 M4 = pred_4[0].cpu().numpy(), ground_truth_M = transform[0])
            
            predict_M_list = []
            for batch_idx, (rgb, warp, transform) in enumerate(loader.load_test_dataset(batch_size = BATCH_SIZE)):
                warp_gpu = warp.to(gpu_dev)
                warp_img = warp[0,:,:,:].numpy()
                warp_img = np.moveaxis(warp_img, -1, 0)
                warp_img = np.moveaxis(warp_img, -1, 0) #for properly displaying image in matplotlib
                
                #print("Showing image for batch ID: ", batch_idx)
                
                pred = cnn(warp.to(gpu_dev))
                predict_M_list.append(pred[0].cpu().numpy())
                
                #for first inference
                reshaped_t = torch.reshape(transform, (np.size(transform, axis = 0), 9)).type('torch.FloatTensor')
                revised_t = torch.reshape(reshaped_t[:,0], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
                loss = loss_func(pred, revised_t)
                accum_loss = accum_loss + loss.cpu().data
                
                #for second inference
                revised_t = torch.reshape(reshaped_t[:,1], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
                pred_2 = cnn_2(warp_gpu)
                loss = loss_func(pred_2, revised_t)
                accum_loss = accum_loss + loss.cpu().data
                
                #for 3rd inference
                revised_t = torch.reshape(reshaped_t[:,3], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
                pred_3 = cnn_3(warp_gpu)
                loss_3 = loss_func(pred_3, revised_t)
                accum_loss = accum_loss + loss.cpu().data
                
                #for 4th inference
                revised_t = torch.reshape(reshaped_t[:,4], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_dev)
                pred_4 = cnn_4(warp_gpu)
                loss_4 = loss_func(pred_4, revised_t)
                accum_loss = accum_loss + loss.cpu().data
                #if((epoch + 1) % 20 != 0): #only save a batch every 25 epochs
                    #break
            
            print("Validation set preview")
            plt.title("Input image")
            plt.imshow(warp_img)
            plt.show()
            show_transform_image(warp_img, M1 = pred[0].cpu().numpy(), M2 = pred_2[0].cpu().numpy(), M3 = pred_3[0].cpu().numpy(),
                                 M4 = pred_4[0].cpu().numpy(), ground_truth_M = transform[0])
            validate_ave_loss = accum_loss / (batch_idx + 1.0)
            writer.add_scalars(CNN_VERSION +'/MSE_loss', {'training_loss' :train_ave_loss, 'validation_loss' : validate_ave_loss}, global_step = (epoch + 1)) #plot validation loss
            writer.close()
            
            print("Current epoch: ", (epoch + 1), " Training loss: ", train_ave_loss, "Validation loss: ", validate_ave_loss)
            if((epoch + 1) % 5 == 0): #only save a batch every X epochs
                gm.save_predicted_transforms(predict_M_list, 0) #use epoch value if want to save per epoch
                torch.save({'epoch': epoch,
                            'cnn_1': cnn.state_dict(),
                            'cnn_1_optimizer_state_dict': optimizer.state_dict(),
                            'cnn_2': cnn_2.state_dict(),
                            'cnn_2_optimizer_state_dict': optimizer_2.state_dict(),
                            'cnn_3': cnn_3.state_dict(),
                            'cnn_3_optimizer_state_dict': optimizer_3.state_dict(),
                            'cnn_4': cnn_4.state_dict(),
                            'cnn_4_optimizer_state_dict': optimizer_4.state_dict()}, CHECKPATH)
      
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_train(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
