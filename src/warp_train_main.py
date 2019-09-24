# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:16:15 2019

Main starting point for warp image training
@author: delgallegon
"""


from visualizers import warp_data_visualizer as visualizer
import torch
from torch.utils.tensorboard import SummaryWriter
from loaders import torch_image_loader as loader
import modular_trainer as trainer

LR = 0.0001
num_epochs = 500
BATCH_SIZE = 32
LAST_STABLE_CNN_VERSION = "cnn_v3.18"
CNN_VERSION = "cnn_v3.18"
OPTIMIZER_KEY = "optimizer"

def start_train(gpu_device):
    #initialize tensorboard writer
    writer = SummaryWriter('train/train_result')
    
    model_list = []
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/1', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/2', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/3', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/4', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/5', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/6', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/7', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    model_list.append(trainer.ModularTrainer(CNN_VERSION + '/8', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = writer, lr = LR))
    
    #checkpoint loading here
    CHECKPATH = 'tmp/' + CNN_VERSION +'.pt'
    start_epoch = 1
    if(False):
        checkpoint = torch.load(CHECKPATH)
        start_epoch = checkpoint['epoch']
        for model in model_list:
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
            #train. NOTE: Model #1 also predicts gt_index = 4
            model_list[0].train(gt_index = 0, current_epoch = epoch, warp = warp, transform = transform)
            model_list[1].train(gt_index = 1, current_epoch = epoch, warp = warp, transform = transform)
            model_list[2].train(gt_index = 2, current_epoch = epoch, warp = warp, transform = transform)
            model_list[3].train(gt_index = 3, current_epoch = epoch, warp = warp, transform = transform)
            model_list[4].train(gt_index = 4, current_epoch = epoch, warp = warp, transform = transform)
            model_list[5].train(gt_index = 5, current_epoch = epoch, warp = warp, transform = transform)
            model_list[6].train(gt_index = 6, current_epoch = epoch, warp = warp, transform = transform)
            model_list[7].train(gt_index = 7, current_epoch = epoch, warp = warp, transform = transform)
            
            for model in model_list:
                accum_loss = accum_loss + model.get_batch_loss()
            
            if(batch_idx % 25 == 0):
                print("Batch id: ", batch_idx, 
                      "\n[",model_list[0].get_name(),"] Loss: ", model_list[0].get_batch_loss(),
                      "\n[",model_list[1].get_name(),"] Loss: ", model_list[1].get_batch_loss(),
                      "\n[",model_list[2].get_name(),"] Loss: ", model_list[2].get_batch_loss(),
                      "\n[",model_list[3].get_name(),"] Loss: ", model_list[3].get_batch_loss(),
                      "\n[",model_list[4].get_name(),"] Loss: ", model_list[4].get_batch_loss(),
                      "\n[",model_list[5].get_name(),"] Loss: ", model_list[5].get_batch_loss(),
                      "\n[",model_list[6].get_name(),"] Loss: ", model_list[6].get_batch_loss(),
                      "\n[",model_list[7].get_name(),"] Loss: ", model_list[7].get_batch_loss())
            
        
        
        #log weights in tensorboard
        for model in model_list:
            model.log_weights(current_epoch = epoch)
        
        train_ave_loss = accum_loss / (len(model_list) * (batch_idx + 1))
        accum_loss = 0.0
        
        #perform inference on training
        warp_img = model_list[-1].get_last_warp_img()
        warp_tensor = model_list[-1].get_last_warp_tensor()
        ground_truth_M = model_list[-1].get_last_transform()
        ground_truth_tensor = model_list[-1].get_last_transform_tensor()
        
        M_list = []
        for model in model_list:
                M, loss = model.single_infer(warp_tensor = warp_tensor, ground_truth_tensor = ground_truth_tensor)
                M_list.append(M)  
        visualizer.show_transform_image(warp_img, M_list = M_list,
                                        ground_truth_M = ground_truth_M,
                                        should_save = False, current_epoch = epoch, save_every_epoch = 5)
        
        
        accum_loss = 0.0
        predict_M_list = []
        
        #perform validation test
        for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
            model_Ms = []
            for model in model_list:
                M, loss = model.batch_infer(warp_tensor = warp, ground_truth_tensor = transform)
                accum_loss = accum_loss + loss
                model_Ms.append(M)
            predict_M_list.append(model_Ms)
        
        M_list = []
        #perform inference on validation
        warp_img = model_list[-1].get_last_warp_img()
        warp_tensor = model_list[-1].get_last_warp_tensor()
        ground_truth_M = model_list[-1].get_last_transform()
        ground_truth_tensor = model_list[-1].get_last_transform_tensor()
        
        for model in model_list:
                M, loss = model.single_infer(warp_tensor = warp_tensor, ground_truth_tensor = ground_truth_tensor)
                M_list.append(M)
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
                for model in model_list:
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

