# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:16:15 2019

Main starting point for CNN warp training
@author: delgallegon
"""


from visualizers import warp_data_visualizer as visualizer
import torch
from torch.utils.tensorboard import SummaryWriter
from loaders import torch_image_loader as loader
import warping_trainer
import numpy as np

LR = 0.0001
num_epochs = 70
BATCH_SIZE = 8
CNN_VERSION = "cnn_v4.23"
CNN_ITERATION = "3"
OPTIMIZER_KEY = "optimizer"

def start_train(gpu_device):
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')
    ct = warping_trainer.WarpingTrainer(CNN_VERSION, gpu_device = gpu_device, writer = writer, lr = LR, weight_decay = 0.0)
     
    #checkpoint loading here
    CHECKPATH = 'tmp/' + CNN_VERSION +'.pt'
    start_epoch = 1
    if(True): 
        checkpoint = torch.load(CHECKPATH)
        start_epoch = checkpoint['epoch'] + 1
        for i in range(ct.model_length):         
            ct.load_saved_states(i,checkpoint[ct.get_name() + str(i)], checkpoint[ct.get_name() + OPTIMIZER_KEY + str(i)])
 
        print("Loaded checkpt ",CHECKPATH, "Current epoch: ", start_epoch)
        print("===================================================")
     
    training_dataset = loader.load_dataset(batch_size = BATCH_SIZE, num_image_to_load = -1)
    test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE, num_image_to_load = 100)
    
    for epoch in range(start_epoch, num_epochs):
        accum_loss = 0.0
        train_ave_loss = 0.0
        val_ave_loss = 0.0
        for batch_idx, (rgb, warp, transform, path) in enumerate(training_dataset):
            ct.train(warp, transform)
            accum_loss = accum_loss + ct.get_batch_loss()
             
            if(batch_idx % 100 == 0):
                print("Batch id: ", batch_idx,
                      "\n[",ct.get_name(),"] Loss: ", ct.get_batch_loss())
            
        
        ct.log_weights(epoch)
        train_ave_loss = accum_loss / (batch_idx + 1)
        
        #perform training inference
        warp_img = ct.get_last_warp_img()
        warp_tensor = ct.get_last_warp_tensor()
        ground_truth_M = ct.get_last_transform()
        ground_truth_tensor = ct.get_last_transform_tensor()
        
        M, loss = ct.infer(warp_tensor, ground_truth_tensor)
        print("Shape: ", np.shape(M), "M: ", M)
        visualizer.show_transform_image(warp_img, M_list = M,
                                    ground_truth_M = ground_truth_M, should_inverse = True,
                                    should_save = False, current_epoch = epoch, save_every_epoch = 5)
        
        accum_loss = 0.0
        #perform validation test
        for batch_idx, (rgb, warp, transform, path) in enumerate(test_dataset):
            M, loss = ct.infer(warp, transform)
            accum_loss = accum_loss + loss
        
        val_ave_loss = accum_loss / (batch_idx +  1)
        
        #perform inference on validation
        warp_img = ct.get_last_warp_img()
        warp_tensor = ct.get_last_warp_tensor()
        ground_truth_M = ct.get_last_transform()
        ground_truth_tensor = ct.get_last_transform_tensor()
        
        M, loss = ct.infer(warp_tensor, ground_truth_tensor)
        visualizer.show_transform_image(warp_img, M_list = M,
                                    ground_truth_M = ground_truth_M, should_inverse = True,
                                    should_save = False, current_epoch = epoch, save_every_epoch = 5)
        
        #train_ave_loss = train_ave_loss.cpu().numpy()
        #val_ave_loss = val_ave_loss.cpu().numpy()
        
        print("Total training loss on epoch ", epoch, ": ", train_ave_loss)
        print("Total validation loss on epoch ", epoch, ": ", val_ave_loss) 
        
        ct.report_new_epoch()
        
        if(epoch != 1): #don't write on first epoch. observation: error too large. skews visualization.
            writer.add_scalars(CNN_VERSION +'/MSE_loss' + "/" + CNN_ITERATION, {'training_loss' :train_ave_loss, 'validation_loss' : val_ave_loss},
                               global_step = epoch) #plot validation loss
            writer.close()
        
        if(epoch % 1 == 0 and epoch != 0): #only save a batch every X epochs
                #visualizer.save_predicted_transforms(predict_M_list, 0) #use epoch value if want to save per epoch
                save_dict = {'epoch': epoch}
                
                for i in range(ct.model_length):
                    model_state_dict, optimizer_state_dict = ct.get_state_dicts(i)
                    save_dict[ct.get_name() + str(i)] = model_state_dict
                    save_dict[ct.get_name() + OPTIMIZER_KEY + str(i)] = optimizer_state_dict
                
                torch.save(save_dict, CHECKPATH)
                print("Saved model state:", len(save_dict))   
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_train(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()

