# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:35:33 2019

Main starting point for testing how good the network is.
Contains inference operations and visualization of results
@author: delgallegon
"""



from visualizers import warp_data_visualizer as warp_visualizer
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import generate_misaligned as gm
from loaders import torch_image_loader as loader
import modular_trainer as trainer
import warp_train_main as train_main
from utils import tensor_utils
from visualizers import results_visualizer
import global_vars as gv
from matplotlib import pyplot as plt


BATCH_SIZE = 32
OPTIMIZER_KEY = "optimizer"
    
def start_test(gpu_device):
    model_list = []
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/1', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 1, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/2', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 2, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/3', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 3, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/4', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 4, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/5', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 5, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/6', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 6, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/7', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 7, lr = train_main.LR))
    model_list.append(trainer.ModularTrainer(train_main.CNN_VERSION + '/8', gpu_device = gpu_device, batch_size = BATCH_SIZE,
                                             writer = None, gt_index = 8, lr = train_main.LR))
    
    #checkpoint loading here
    CHECKPATH = 'tmp/' + train_main.LAST_STABLE_CNN_VERSION +'.pt'
    checkpoint = torch.load(CHECKPATH)
    for model in model_list:
        model.load_saved_states(checkpoint[model.get_name()], checkpoint[model.get_name() + OPTIMIZER_KEY])
 
    print("Loaded checkpt ",CHECKPATH)
    
    test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE, full_infer = True)
    #perform inference on batches
#    overall_index = 0;
#    for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
#        print("Batch idx: ", batch_idx)
#        for index in range(len(warp)):
#            model_Ms = [];
#            for model in model_list:
#                warp_candidate = torch.unsqueeze(warp[index,:,:,:],  0).to(gpu_device)
#                reshaped_t = torch.reshape(transform[index], (1, 9)).type('torch.FloatTensor')
#                gt_candidate = torch.reshape(reshaped_t[:,model.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_device)
#        
#                M, loss = model.single_infer(warp_tensor = warp_candidate, ground_truth_tensor = gt_candidate)
#                model_Ms.append(M)
#            
#            #chance visualize and save result
#            warp_img = tensor_utils.convert_to_matplotimg(warp, index)
#            rgb_img = tensor_utils.convert_to_matplotimg(rgb, index)
#            warp_visualizer.visualize_results(warp_img = warp_img, rgb_img = rgb_img, M_list = model_Ms, ground_truth_M = transform[index], index = overall_index)
#            overall_index = overall_index + 1
                    
    #visualize each layer's output
#    for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
#        #get first data per batch only
#        model_Ms = [];
#        for model in model_list:
#            warp_candidate = torch.unsqueeze(warp[0,:,:,:], 0).to(gpu_device)
#            reshaped_t = torch.reshape(transform[0], (1, 9)).type('torch.FloatTensor')
#            gt_candidate = torch.reshape(reshaped_t[:,model.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_device)
#    
#            M, loss = model.single_infer(warp_tensor = warp_candidate, ground_truth_tensor = gt_candidate)
#            model_Ms.append(M)
#        
#        conv_activation, pool_activation = model_list[0].get_model_layer(3)
#        results_visualizer.visualize_layer(conv_activation, filter_range = 48, resize_scale = 2)
    
    measure_performance(gpu_device, model_list, test_dataset)

def compute_dataset_mean(model_list, test_dataset):
    accumulate_T = np.zeros(9)
    count = 0     
    for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
        for index in range(len(warp)):
            reshaped_t = torch.reshape(transform[index], (1, 9)).type('torch.FloatTensor')
            accumulate_T = accumulate_T + reshaped_t.numpy()
            count = count + 1
             
    dataset_mean = accumulate_T / count * 1.0
    np.savetxt(gv.IMAGE_PATH_PREDICT + "dataset_mean.txt", dataset_mean)

def measure_performance(gpu_device,model_list, test_dataset):
    dataset_mean = np.loadtxt(gv.IMAGE_PATH_PREDICT + "dataset_mean.txt")
    print("Dataset mean is: ", dataset_mean)
    
    count = 0
    failures = 0
    accum_mae = [0.0, 0.0, 0.0] #dataset mean, homography, our method
    average_MAE = [0.0, 0.0, 0.0] 
    accum_mse = [0.0, 0.0, 0.0]
    average_MSE = [0.0, 0.0, 0.0]
    average_RMSE = [0.0, 0.0, 0.0]
    
    for batch_idx, (rgb, warp, transform) in enumerate(test_dataset):
        for i in range(np.shape(warp)[0]):
            model_Ms = [];
            warp_candidate = torch.unsqueeze(warp[i,:,:,:], 0).to(gpu_device)
            reshaped_t = torch.reshape(transform[i], (1, 9)).type('torch.FloatTensor')
            for model in model_list:
                gt_candidate = torch.reshape(reshaped_t[:,model.gt_index], (np.size(reshaped_t, axis = 0), 1)).type('torch.FloatTensor').to(gpu_device)
                M, loss = model.single_infer(warp_tensor = warp_candidate, ground_truth_tensor = gt_candidate)
                model_Ms.append(M)
        
            model_Ms.append(1.0) #append 1.0 as element M[2,2]
            warp_img = tensor_utils.convert_to_matplotimg(warp, i)
            rgb_img = tensor_utils.convert_to_matplotimg(rgb, i)
            homog_img, homography_M = warp_visualizer.warp_perspective_least_squares(warp_img, rgb_img)
            
            h_intensity = np.sum(np.absolute(homography_M - transform[i].numpy()))
            if(h_intensity > 100):
                failures = failures + 1
#                plt.imshow(homog_img)
#                plt.savefig(gv.IMAGE_PATH_PREDICT + "outlier_h_"+str(count)+".png")
#                plt.show()
                h_intensity = 0 #simply set it to 0 so it won't have any effect in the results
            
            #measure SSIM
            matrix_mean = np.reshape(dataset_mean, (3,3))
            matrix_own = np.reshape(model_Ms, (3,3))
            SSIM = warp_visualizer.measure_ssim(warp_img, rgb_img, matrix_mean, homography_M, matrix_own)
            
            accum_mae[0] = accum_mae[0] + np.absolute(dataset_mean - reshaped_t.numpy())
            accum_mae[1] = accum_mae[1] + np.absolute(h_intensity)
            accum_mae[2] = accum_mae[2] + np.absolute(model_Ms - reshaped_t.numpy())
            
            accum_mse[0] = accum_mse[0] + np.power(np.absolute(dataset_mean - reshaped_t.numpy()),2)
            accum_mse[1] = accum_mse[1] + np.power(h_intensity,2)
            accum_mse[2] = accum_mse[2] + np.power(np.absolute(model_Ms - reshaped_t.numpy()),2)
            
            count = count + 1
        
        #print("Batch id: ", batch_idx, "Count: ", count)
    
    average_MAE[0] = np.round(np.sum(accum_mae[0] / (count * 1.0)), 4)
    average_MAE[1] = np.round(np.sum(accum_mae[1] / (count * 1.0)), 4)
    average_MAE[2] = np.round(np.sum(accum_mae[2] / (count * 1.0)), 4)
    
    average_MSE[0] = np.round(np.sum(accum_mse[0] / (count * 1.0)), 4)
    average_MSE[1] = np.round(np.sum(accum_mse[1] / (count * 1.0)), 4)
    average_MSE[2] = np.round(np.sum(accum_mse[2] / (count * 1.0)), 4)
    
    average_RMSE[0] = np.round(np.sqrt(average_MSE[0]), 4)
    average_RMSE[1] = np.round(np.sqrt(average_MSE[1]), 4)
    average_RMSE[2] = np.round(np.sqrt(average_MSE[2]), 4)
    
    print("Average MAE using dataset mean: ", average_MAE[0])
    print("Average MAE using homography estimation: ", average_MAE[1])
    print("Average MAE using our method: ", average_MAE[2])
    
    print("Average MSE using dataset mean: ", average_MSE[0])
    print("Average MSE using homography estimation: ", average_MSE[1])
    print("Average MSE using our method: ", average_MSE[2])
    
    print("Average RMSE using dataset mean: ", average_RMSE[0])
    print("Average RMSE using homography estimation: ", average_RMSE[1])
    print("Average RMSE using our method: ", average_RMSE[2])
    
    failure_rate = np.round((failures / (count * 1.0)),4)
    print("Homography failure rate: ", failure_rate)
                
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_test(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
