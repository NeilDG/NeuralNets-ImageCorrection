# -*- coding: utf-8 -*-
"""
Class for training an autoencoder
Created on Thu Oct 10 20:37:53 2019

@author: delgallegon
"""

from visualizers import warp_data_visualizer as visualizer
import torch
from torch.utils.tensorboard import SummaryWriter
from loaders import torch_image_loader as loader
import modular_autoencoder_trainer as ae
import numpy as np

LR = 0.001
num_epochs = 65
BATCH_SIZE = 32
LAST_STABLE_AE_VERSION = "ae_v4.00"
AE_VERSION = "ae_v4.00"
OPTIMIZER_KEY = "optimizer"

def start_train(device):
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')

    ae_trainer = ae.AutoEncoderTrainer(AE_VERSION, gpu_device = device, batch_size = BATCH_SIZE, writer = writer, lr = LR)
    
    #checkpoint loading here
    CHECKPATH = 'tmp/' + AE_VERSION +'.pt'
    start_epoch = 1
    if(False):
        checkpoint = torch.load(CHECKPATH)
        start_epoch = checkpoint['epoch']
        
        ae_trainer.load_saved_states(checkpoint[ae_trainer.get_name()], checkpoint[ae_trainer.get_name() + OPTIMIZER_KEY])
        
        print("Loaded checkpt ",CHECKPATH, "Current epoch: ", start_epoch)
        print("===================================================")
    
    training_dataset = loader.load_dataset(batch_size = BATCH_SIZE, fast_train = False)
    test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE)
        
    for epoch in range(start_epoch, num_epochs):
        accum_loss = 0.0
        train_ave_loss = 0.0
        val_ave_loss = 0.0
        for batch_idx, (rgb, warp, transform) in enumerate(training_dataset):
            ae_trainer.train(input_img = warp, ground_truth_img = rgb, current_epoch = epoch)
            accum_loss = accum_loss + ae_trainer.get_batch_loss()
            
            if(batch_idx % 25 == 0):
                print("Batch id: ", batch_idx, 
                      "\n[",ae_trainer.get_name(),"] Loss: ", ae_trainer.get_batch_loss())
            
        #log weights in tensorboard
        ae_trainer.log_weights(epoch)
        
        train_ave_loss = accum_loss / (batch_idx + 1)
        accum_loss = 0.0
        
        #perform inference on training
        warp_tensor = ae_trainer.get_last_warp_tensor()
        warp_img = ae_trainer.get_last_warp_img()
        ground_truth_tensor = ae_trainer.get_last_ground_truth_tensor()
        ground_truth_img = ae_trainer.get_last_ground_truth_img()
        pred_img, loss = ae_trainer.single_infer(warp_tensor, ground_truth_tensor)
        pred_img = np.moveaxis(pred_img, -1, 0)
        pred_img = np.moveaxis(pred_img, -1, 0) #for properly displaying image in matplotlib
        visualizer.show_auto_encoder_img(warp_img, pred_img, ground_truth_img, test_title = "Training")
        
        #perform validation test
        for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
            M, loss = ae_trainer.batch_infer(warp, rgb)
            accum_loss = accum_loss + loss
            
        #perform inference on validation
        warp_tensor = ae_trainer.get_last_warp_tensor()
        warp_img = ae_trainer.get_last_warp_img()
        ground_truth_tensor = ae_trainer.get_last_ground_truth_tensor()
        ground_truth_img = ae_trainer.get_last_ground_truth_img()
        pred_img, loss = ae_trainer.single_infer(warp_tensor, ground_truth_tensor)
        pred_img = np.moveaxis(pred_img, -1, 0)
        pred_img = np.moveaxis(pred_img, -1, 0) #for properly displaying image in matplotlib
        #visualizer.show_auto_encoder_img(warp_img, pred_img, ground_truth_img, test_title = "Validation") TODO: STILL ERROR IN VISUALIZING IMAGES IN VALIDATION
        
        val_ave_loss = accum_loss / (batch_idx + 1)
        print("Total training loss on epoch ", epoch, ": ", train_ave_loss)
        print("Total validation loss on epoch ", epoch, ": ", val_ave_loss) 
        
        writer.add_scalars(AE_VERSION +'/MSE_loss', {'training_loss' :train_ave_loss, 'validation_loss' : val_ave_loss},
                           global_step = epoch) #plot validation loss
        writer.close()
        
        if(epoch % 1 == 0 and epoch != 0): #only save a batch every X epochs
                save_dict = {'epoch': epoch}
                model_state_dict, optimizer_state_dict = ae_trainer.get_state_dicts()
                save_dict[ae_trainer.get_name()] = model_state_dict
                save_dict[ae_trainer.get_name() + OPTIMIZER_KEY] = optimizer_state_dict
                
                torch.save(save_dict, CHECKPATH)
                print("Saved model state:", len(save_dict))
        
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_train(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()