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
from model import warp_cnn_joiner
import modular_trainer as trainer
import intermediate_trainer
import numpy as np

LR = 0.001
num_epochs = 65
BATCH_SIZE = 8
CNN_VERSION = "cnn_v3.25"
OPTIMIZER_KEY = "optimizer"

def start_train(gpu_device):
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')
    
    intermediate_list = []
    intermediate_list.append(intermediate_trainer.IntermediateTrainer(CNN_VERSION + 'im/2', gpu_device = gpu_device, writer = writer))
    intermediate_list.append(intermediate_trainer.IntermediateTrainer(CNN_VERSION + 'im/3', gpu_device = gpu_device, writer = writer))
    intermediate_list.append(intermediate_trainer.IntermediateTrainer(CNN_VERSION + 'im/4', gpu_device = gpu_device, writer = writer))
    intermediate_list.append(intermediate_trainer.IntermediateTrainer(CNN_VERSION + 'im/6', gpu_device = gpu_device, writer = writer))
    intermediate_list.append(intermediate_trainer.IntermediateTrainer(CNN_VERSION + 'im/7', gpu_device = gpu_device, writer = writer))
    intermediate_list.append(intermediate_trainer.IntermediateTrainer(CNN_VERSION + 'im/8', gpu_device = gpu_device, writer = writer))
    
    model_list = []
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/2', gpu_device = gpu_device, writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/3', gpu_device = gpu_device, writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/4', gpu_device = gpu_device, writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/6', gpu_device = gpu_device, writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/7', gpu_device = gpu_device, writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/8', gpu_device = gpu_device, writer = writer, lr = LR))
    
    #checkpoint loading here
    CHECKPATH = 'tmp/' + CNN_VERSION +'.pt'
    start_epoch = 1
    if(False):
        checkpoint = torch.load(CHECKPATH)
        start_epoch = checkpoint['epoch']
        for intermediate, model in zip(intermediate_list,model_list):
            intermediate.load_saved_states(checkpoint[model.get_name()])
            model.load_saved_states(checkpoint[model.get_name()], checkpoint[model.get_name() + OPTIMIZER_KEY])
 
        print("Loaded checkpt ",CHECKPATH, "Current epoch: ", start_epoch)
        print("===================================================")
     
    training_dataset = loader.load_dataset(batch_size = BATCH_SIZE, fast_train = False)
    test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE)
    
    for epoch in range(start_epoch, num_epochs):
        accum_loss = 0.0
        train_ave_loss = 0.0
        val_ave_loss = 0.0
        for batch_idx, (rgb, warp, transform) in enumerate(training_dataset):
            for intermediate, model in zip(intermediate_list,model_list):
                maps = []
                maps.append(intermediate_list[0].produce_intermediate(gt_index = 1,warp = warp, transform = transform))
                maps.append(intermediate_list[1].produce_intermediate(gt_index = 2, warp = warp, transform = transform))
                maps.append(intermediate_list[2].produce_intermediate(gt_index = 3, warp = warp, transform = transform))
                maps.append(intermediate_list[3].produce_intermediate(gt_index = 5, warp = warp, transform = transform))
                maps.append(intermediate_list[4].produce_intermediate(gt_index = 6, warp = warp, transform = transform))
                maps.append(intermediate_list[5].produce_intermediate(gt_index = 7, warp = warp, transform = transform))
                model.train(gt_index = intermediate.get_gt_index(), 
                           intermediate_tensor = maps, transform = transform)
                accum_loss = accum_loss + model.get_batch_loss()
            
            if(batch_idx % 25 == 0):
                print("Batch id: ", batch_idx, 
                      "\n[",model_list[0].get_name(),"] Loss: ", model_list[0].get_batch_loss(),
                      "\n[",model_list[1].get_name(),"] Loss: ", model_list[1].get_batch_loss(),
                      "\n[",model_list[2].get_name(),"] Loss: ", model_list[2].get_batch_loss(),
                      "\n[",model_list[3].get_name(),"] Loss: ", model_list[3].get_batch_loss(),
                      "\n[",model_list[4].get_name(),"] Loss: ", model_list[4].get_batch_loss(),
                      "\n[",model_list[5].get_name(),"] Loss: ", model_list[5].get_batch_loss())
        
        #log weights in tensorboard
        for intermediate, model in zip(intermediate_list,model_list):
            intermediate.log_weights(epoch)
            model.log_weights(epoch)
        
        train_ave_loss = accum_loss / (len(model_list) * (batch_idx + 1))
        accum_loss = 0.0
        
        #perform inference on training
        warp_img = intermediate_list[-1].get_last_warp_img()
        warp_tensor = intermediate_list[-1].get_last_warp_tensor()
        ground_truth_M = intermediate_list[-1].get_last_transform()
        ground_truth_tensor = intermediate_list[-1].get_last_transform_tensor()
        
        M_list = []
        for model in model_list:
            maps = []
            maps.append(intermediate_list[0].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[1].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[2].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[3].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[4].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[5].single_infer(warp_tensor, transform))
            M, loss = model.single_infer(maps, ground_truth_tensor)
            M_list.append(M)  
        print("Training inference")
        visualizer.show_transform_image(warp_img, M_list = M_list,
                                        ground_truth_M = ground_truth_M,
                                        should_save = False, current_epoch = epoch, save_every_epoch = 5)
        
        
        accum_loss = 0.0
        predict_M_list = []
        
        #perform validation test
        for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
            model_Ms = []
            for model in model_list:
                maps = []
                maps.append(intermediate_list[0].batch_infer(warp_tensor, transform))
                maps.append(intermediate_list[1].batch_infer(warp_tensor, transform))
                maps.append(intermediate_list[2].batch_infer(warp_tensor, transform))
                maps.append(intermediate_list[3].batch_infer(warp_tensor, transform))
                maps.append(intermediate_list[4].batch_infer(warp_tensor, transform))
                maps.append(intermediate_list[5].batch_infer(warp_tensor, transform))
                M, loss = model.batch_infer(maps, transform)
                accum_loss = accum_loss + loss
                model_Ms.append(M)
            predict_M_list.append(model_Ms)
        
        M_list = []
        #perform inference on validation
        warp_img = intermediate_list[-1].get_last_warp_img()
        warp_tensor = intermediate_list[-1].get_last_warp_tensor()
        ground_truth_M = intermediate_list[-1].get_last_transform()
        ground_truth_tensor = intermediate_list[-1].get_last_transform_tensor()
        
        for model in model_list:
            maps = []
            maps.append(intermediate_list[0].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[1].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[2].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[3].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[4].single_infer(warp_tensor, transform))
            maps.append(intermediate_list[5].single_infer(warp_tensor, transform))
            M, loss = model.single_infer(maps, ground_truth_tensor)
            M_list.append(M)
        print("Validation inference")
        visualizer.show_transform_image(warp_img, M_list = M_list,
                                        ground_truth_M = ground_truth_M,
                                        should_save = False, current_epoch = epoch, save_every_epoch = 3)
        
        val_ave_loss = accum_loss / (len(model_list) * (batch_idx + 1))
        print("Total training loss on epoch ", epoch, ": ", train_ave_loss)
        print("Total validation loss on epoch ", epoch, ": ", val_ave_loss) 
        
        writer.add_scalars(CNN_VERSION +'/MSE_loss', {'training_loss' :train_ave_loss, 'validation_loss' : val_ave_loss},
                           global_step = epoch) #plot validation loss
        writer.close()
        
        if(epoch % 1 == 0 and epoch != 0): #only save a batch every X epochs
                visualizer.save_predicted_transforms(predict_M_list, 0) #use epoch value if want to save per epoch
                save_dict = {'epoch': epoch}
                for intermediate, model in zip(intermediate_list,model_list):
                    model_state_dict = intermediate.get_state_dicts()
                    save_dict[model.get_name()] = model_state_dict
                    
                    model_state_dict, optimizer_state_dict = model.get_state_dicts()
                    save_dict[model.get_name()] = model_state_dict
                    save_dict[model.get_name() + OPTIMIZER_KEY] = optimizer_state_dict
                
                torch.save(save_dict, CHECKPATH)
                print("Saved model state:", len(save_dict))

    
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_train(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()

