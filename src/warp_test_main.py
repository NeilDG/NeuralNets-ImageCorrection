# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:35:33 2019

Main starting point for testing how good the network is.
Contains inference operations and visualization of results
@author: delgallegon
"""



from visualizers import warp_data_visualizer as warp_visualizer
import torch
import cv2
import numpy as np
from loaders import torch_image_loader as loader
import warping_trainer
import warp_train_main as train_main
from utils import tensor_utils
import global_vars as gv
from matplotlib import pyplot as plt

BATCH_SIZE = 32
OPTIMIZER_KEY = "optimizer"

def start_test(gpu_device):
    ct = warping_trainer.WarpingTrainer(train_main.CNN_VERSION, gpu_device = gpu_device, writer = None, lr = train_main.LR)
    
    #checkpoint loading here
    CHECKPATH = 'tmp/' + train_main.CNN_VERSION +'.pt'
    checkpoint = torch.load(CHECKPATH)
    for i in range(ct.model_length):         
        ct.load_saved_states(i,checkpoint[ct.get_name() + str(i)], checkpoint[ct.get_name() + OPTIMIZER_KEY + str(i)])
 
    print("Loaded checkpt ",CHECKPATH)
    
    test_dataset = loader.load_test_dataset(batch_size = BATCH_SIZE, num_image_to_load = 2000)
    #compute_dataset_mean(test_dataset)
    measure_performance(gpu_device, ct, test_dataset)
    
def compute_dataset_mean(test_dataset):
    accumulate_T = np.zeros(9)
    count = 0   
     
    for batch_idx, (rgb, warp, transform, path) in enumerate(test_dataset):
        for index in range(len(warp)):
            reshaped_t = torch.reshape(transform[index], (1, 9)).type('torch.FloatTensor')
            accumulate_T = accumulate_T + reshaped_t.numpy()
            count = count + 1
             
    dataset_mean = accumulate_T / count * 1.0
    np.savetxt(gv.IMAGE_PATH_PREDICT + "dataset_mean.txt", dataset_mean, fmt = "%.8f")
    
def measure_performance(gpu_device, trainer, test_dataset):
    dataset_mean = np.loadtxt(gv.IMAGE_PATH_PREDICT + "dataset_mean.txt")
    print("Dataset mean is: ", dataset_mean)
    
    count = 0
    failures = 0
    accum_mae = [0.0, 0.0, 0.0, 0.0] #dataset mean, homography, RRL 1, our method
    average_MAE = [0.0, 0.0, 0.0, 0.0] 
    accum_mse = [0.0, 0.0, 0.0, 0.0]
    average_MSE = [0.0, 0.0, 0.0, 0.0]
    average_RMSE = [0.0, 0.0, 0.0, 0.0]
    
    accum_ssim = [0.0, 0.0, 0.0, 0.0, 0.0]
    average_SSIM = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    pixel_mse = [0.0, 0.0, 0.0, 0.0, 0.0]
    pixel_rmse = [0.0, 0.0, 0.0, 0.0, 0.0]
    average_pixel_MSE = [0.0, 0.0, 0.0, 0.0, 0.0]
    average_pixel_RMSE = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    M_list = []
    for batch_idx, (rgb, warp, transform, path) in enumerate(test_dataset):
        for i in range(np.shape(warp)[0]):
            warp_candidate = torch.unsqueeze(warp[i,:,:,:], 0)
            reshaped_t = torch.reshape(transform[i], (1, 9)).type('torch.FloatTensor')
            M, loss = trainer.infer(warp_candidate, reshaped_t)
            
            #append element on correct places
            ground_truth_M = np.ndarray.flatten(reshaped_t.numpy())
            #print("Ground truth M shape:", ground_truth_M)
            
            M = np.insert(M, 2, 0.0)
            M = np.insert(M, 5, 0.0)
            M = np.append(M, 1.0)
            M_list.append(M)
            #print("Predicted M: ", M)
            #print("Actual M: ", reshaped_t.numpy())
            warp_img = tensor_utils.convert_to_matplotimg(warp, i)
            rgb_img = tensor_utils.convert_to_matplotimg(rgb, i)
            homog_img, homography_M = warp_visualizer.warp_perspective_least_squares(warp_img, rgb_img)
            
            h_intensity = np.sum(np.absolute(homography_M - transform[i].numpy()))
            if(h_intensity > 10):
                failures = failures + 1
                h_intensity = 0 #simply set it to 0 so it won't have any effect in the results
             
            accum_mae[0] = accum_mae[0] + np.round(np.absolute(dataset_mean - reshaped_t.numpy()), 8)
            accum_mae[1] = accum_mae[1] + np.round(np.absolute(h_intensity), 8)
            accum_mae[2] = accum_mae[2] + np.round(np.absolute(M - reshaped_t.numpy()) , 8)
            
            accum_mse[0] = accum_mse[0] + np.round(np.power(np.absolute(dataset_mean - reshaped_t.numpy()),2), 8)
            accum_mse[1] = accum_mse[1] + np.round(np.power(h_intensity,2), 8)
            accum_mse[2] = accum_mse[2] + np.power(np.absolute(M - reshaped_t.numpy()),2)
            
            #measure SSIM
            matrix_mean = np.reshape(dataset_mean, (3,3))
            matrix_own = np.reshape(M, (3,3))
            chance = np.random.rand() * 100
            
            image_name = path[i].split(".")[0]
            rrl_1_path = gv.RRL_RESULTS_PATH[0] + image_name + "_m" + ".jpg"
            rrl_img_1 = tensor_utils.load_image(rrl_1_path)
            
            rrl_2_path = gv.RRL_RESULTS_PATH[1] + image_name + "_r" + ".png"
            rrl_img_2 = tensor_utils.load_image(rrl_2_path)
            
            SSIM, MSE, RMSE = warp_visualizer.measure_with_rrl(image_name, warp_img, rrl_img_1, rrl_img_2, rgb_img, matrix_mean, homography_M, matrix_own, count, should_visualize = (chance < 30))
            print("Img ", count, " SSIM: ", SSIM, "Chance: ", chance)
            
            for i in range(len(accum_ssim)):
                accum_ssim[i] = accum_ssim[i] + SSIM[i]
                pixel_mse[i] = pixel_mse[i] + MSE[i]
                pixel_rmse[i] = pixel_rmse[i] + RMSE[i]
                
            count = count + 1
            # try:
                
            # except:
            #     print("RRL path possible error: ", rrl_1_path)
    
    warp_visualizer.save_predicted_transforms(M_list)
    average_MAE[0] = np.round(np.sum(accum_mae[0] / (count * 1.0)), 16)
    average_MAE[1] = np.round(np.sum(accum_mae[1] / (count * 1.0)), 16)
    average_MAE[2] = np.round(np.sum(accum_mae[2] / (count * 1.0)), 16)
    
    average_MSE[0] = np.round(np.sum(accum_mse[0] / (count * 1.0)), 16)
    average_MSE[1] = np.round(np.sum(accum_mse[1] / (count * 1.0)), 16)
    average_MSE[2] = np.round(np.sum(accum_mse[2] / (count * 1.0)), 16)
    
    average_RMSE[0] = np.round(np.sqrt(average_MSE[0]), 16)
    average_RMSE[1] = np.round(np.sqrt(average_MSE[1]), 16)
    average_RMSE[2] = np.round(np.sqrt(average_MSE[2]), 16)
     
    for i in range(len(accum_ssim)):
        average_SSIM[i] = np.round(accum_ssim[i] / (count * 1.0), 8)
        average_pixel_MSE[i] = np.round(np.sum(pixel_mse[i] / (count * 1.0)), 8)
        average_pixel_RMSE[i] = np.round(np.sum(pixel_rmse[i] / (count * 1.0)), 8)
        
    with open(gv.IMAGE_PATH_PREDICT + "test_data_result.txt", "w") as f:
        print("Average MAE using dataset mean: ", average_MAE[0], file = f)
        print("Average MAE using homography estimation: ", average_MAE[1], file = f)
        print("Average MAE using our method: ", average_MAE[2], file = f)
        print("")
        
        print("Average MSE using dataset mean: ", average_MSE[0], file = f)
        print("Average MSE using homography estimation: ", average_MSE[1], file = f)
        print("Average MSE using our method: ", average_MSE[2], file = f)
        print("")
        
        print("Average RMSE using dataset mean: ", average_RMSE[0], file = f)
        print("Average RMSE using homography estimation: ", average_RMSE[1], file = f)
        print("Average RMSE using our method: ", average_RMSE[2], file = f)
        print("")
        
        print("Average pixel MSE using dataset mean: ", average_pixel_MSE[0], file = f)
        print("Average pixel MSE using homography estimation: ", average_pixel_MSE[1], file = f)
        print("Average pixel MSE using RRL 1: ", average_pixel_MSE[2], file = f)
        print("Average pixel MSE using RRL 2: ", average_pixel_MSE[3], file = f)
        print("Average pixel MSE using our method: ", average_pixel_MSE[4], file = f)
        print("")
        
        print("Average pixel RMSE using dataset mean: ", average_pixel_RMSE[0], file = f)
        print("Average pixel RMSE using homography estimation: ", average_pixel_RMSE[1], file = f)
        print("Average pixel RMSE using RRL 1: ", average_pixel_RMSE[2], file = f)
        print("Average pixel RMSE using RRL 2: ", average_pixel_RMSE[3], file = f)
        print("Average pixel RMSE using our method: ", average_pixel_RMSE[4], file = f)
        print("")
        
        print("Average SSIM using dataset mean: ", average_SSIM[0], file = f)
        print("Average SSIM using homography estimation: ", average_SSIM[1], file = f)
        print("Average SSIM using RRL 1: ", average_SSIM[2], file = f)
        print("Average SSIM using RRL 2: ", average_SSIM[3], file = f)
        print("Average SSIM using our method: ", average_SSIM[4], file = f)
        
        failure_rate = np.round((failures / (count * 1.0)),4)
        print("Homography failure rate: ", failure_rate, file = f)
 

def check_on_unseen_data(gpu_device, model_list):
    unseen_dataset = loader.load_unseen_dataset(BATCH_SIZE)
    
    overall_index = 0;
    for batch_idx, (rgb, warp, transform) in enumerate(unseen_dataset):
        print("Batch idx: ", batch_idx)
        for index in range(len(warp)):
            model_Ms = [];
            for model in model_list:
                warp_candidate = torch.unsqueeze(warp[index,:,:,:],  0).to(gpu_device)
                M = model.blind_infer(warp_tensor = warp_candidate)
                model_Ms.append(M)
            
            #chance visualize and save result
            warp_img = tensor_utils.convert_to_matplotimg(warp, index)
            rgb_img = tensor_utils.convert_to_matplotimg(rgb, index)
            warp_visualizer.visualize_blind_results(warp_img = warp_img, rgb_img = rgb_img, 
                                                    M_list = model_Ms, index = overall_index, 
                                                    p = 1.0)
            overall_index = overall_index + 1
               
def main():
    if(torch.cuda.is_available()) :
        print("NVIDIA CUDA is ready! ^_^")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    start_test(device)
    
    

if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()
