# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 11:47:47 2019
Data visualizer for analyzing input data
@author: delgallegon
"""
from loaders import torch_image_loader as loader
import os
import numpy as np
import cv2
import global_vars as gv
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim
from skimage.measure import compare_mse
from skimage.measure import compare_nrmse

#saves predicted transforms inferred by network. Always set start_index = 0 if you want to
#override saved predictions
def save_predicted_transforms(M_list, start_index = 0):
    for i in range(np.shape(M_list)[0]):
        np.savetxt(gv.SAVE_PATH_PREDICT + "warp_" +str(i + start_index)+ ".txt", M_list[i])
        print("Successfully saved predicted M ", str(i + start_index))

def retrieve_predict_warp_list():
    warp_list = [];
    
    for (dirpath, dirnames, filenames) in os.walk(gv.SAVE_PATH_PREDICT):
        for f in filenames:
            if f.endswith(".txt"):
                warp_list.append(os.path.join(dirpath, f))
    
    return warp_list

def hide_plot_legend(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # X AXIS -BORDER
    ax.spines['bottom'].set_visible(False)
    # BLUE
    ax.set_xticklabels([])
    # RED
    ax.set_xticks([])
    # RED AND BLUE TOGETHER
    ax.axes.get_xaxis().set_visible(False)
    
    # Y AXIS -BORDER
    ax.spines['left'].set_visible(False)
    # YELLOW
    ax.set_yticklabels([])
    # GREEN
    ax.set_yticks([])
    # YELLOW AND GREEN TOGHETHER
    ax.axes.get_yaxis().set_visible(False)

#performs perspective transformation by least squares
def warp_perspective_least_squares(warp_img, rgb_img):
    
    warp_img = np.moveaxis(warp_img, -3, 0); warp_img = np.uint8(warp_img * 255)
    rgb_img = np.moveaxis(rgb_img, -3, 0); rgb_img = np.uint8(rgb_img * 255)
    
    im1Gray = cv2.cvtColor(warp_img, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(700)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
  
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    #remove not so good matches
    good_match_threshold = 0.20
    numGoodMatches = int(len(matches) * good_match_threshold)
    matches = matches[:numGoodMatches]
  
    # Draw top matches
    im_matches = cv2.drawMatches(warp_img, keypoints1, rgb_img, keypoints2, matches, None)
  
    # Extract location of good matches
    points1 = np.zeros((len(im_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(im_matches), 2), dtype=np.float32)
 
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
   
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    #print("Resulting H:" , h, np.shape(h), "Key points shape: ", np.shape(keypoints1), np.shape(keypoints2))
    # Use homography
    height, width, channels = rgb_img.shape
    if(np.shape(h) == (3,3)):
        result_img = cv2.warpPerspective(warp_img, h, (gv.WARP_W, gv.WARP_H),
                                         borderValue = (255,255,255))
    else:
        print("H is not 3x3!")
        h = np.ones((3,3))
        result_img = warp_img
    im1Gray = None
    im2Gray = None
    im_matches = None
    points1 = None
    points2 = None

    return result_img, h

def show_blind_image_test(rgb, least_squares_img, M_list, ground_truth_img, index, should_save):
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    f.set_size_inches(12,13)
    
    hide_plot_legend(ax1)
    #ax1.set_title("Input")
    ax1.imshow(rgb)
    
    pred_M = np.ones((3,3))
    pred_M[0,0] = M_list[0]
    pred_M[0,1] = M_list[1]
    pred_M[0,2] = M_list[2]
    pred_M[1,0] = M_list[3]
    pred_M[1,1] = M_list[4]
    pred_M[1,2] = M_list[5]
    pred_M[2,0] = M_list[6]
    pred_M[2,1] = M_list[7]
    
    hide_plot_legend(ax2)
    #ax2.set_title("Least squares warp")
    ax2.imshow(least_squares_img)
    
    result = cv2.warpPerspective(rgb, pred_M, (np.shape(rgb)[1], np.shape(rgb)[0]))
    hide_plot_legend(ax3)
    #ax3.set_title("Predicted warp")
    ax3.imshow(result)
    
    #ground_truth_img = cv2.resize(ground_truth_img, (gv.WARP_W, gv.WARP_H), interpolation = cv2.INTER_CUBIC)  # scale image up
    #hide_plot_legend(ax4)
    #ax4.set_title("Ground truth")
    #ax4.imshow(ground_truth_img)
    
    if(should_save):
        plt.savefig(gv.IMAGE_PATH_PREDICT + "/result_"+str(index)+ ".png", bbox_inches='tight', pad_inches=0)
    
    plt.show()
    plt.close()
    
def show_transform_image_test(rgb, least_squares_img, M_list, ground_truth_M, should_save, index):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    f.set_size_inches(12,10)
    
    hide_plot_legend(ax1)
    #ax1.set_title(title)
    ax1.imshow(rgb)
    
    pred_M = np.ones((3,3))
    pred_M[0,0] = M_list[0]
    pred_M[0,1] = M_list[1]
    pred_M[0,2] = M_list[2]
    pred_M[1,0] = M_list[3]
    pred_M[1,1] = M_list[4]
    pred_M[1,2] = M_list[5]
    pred_M[2,0] = M_list[6]
    pred_M[2,1] = M_list[7]
    
    result = cv2.warpPerspective(rgb, ground_truth_M.numpy(), (np.shape(rgb)[1], np.shape(rgb)[0]), 
                                 borderValue = (1,1,1))
    hide_plot_legend(ax4)
    #ax2.set_title("Ground truth")
    ax4.imshow(result)
    
    hide_plot_legend(ax2)
    #ax2.set_title("Least squares warp")
    ax2.imshow(least_squares_img)
    
    result = cv2.warpPerspective(rgb, pred_M, (np.shape(rgb)[1], np.shape(rgb)[0]),
                                 borderValue = (1,1,1))
    hide_plot_legend(ax3)
    #ax3.set_title("Predicted warp")
    ax3.imshow(result)
    
    if(should_save):
        plt.savefig(gv.IMAGE_PATH_PREDICT + "/result_"+str(index)+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()
    
    print("Input size: ", np.shape(rgb), "Predict size: ", np.shape(result), " Least squares size: ", np.shape(least_squares_img))
    
    print("Predicted M1 val: ", M_list[0], "Actual val: ",ground_truth_M[0,0].numpy())
    print("Predicted M2 val: ", M_list[1], "Actual val: ",ground_truth_M[0,1].numpy())
    print("Predicted M3 val: ", M_list[2], "Actual val: ",ground_truth_M[0,2].numpy())
    print("Predicted M4 val: ", M_list[3], "Actual val: ",ground_truth_M[1,0].numpy())
    print("Predicted M5 val: ", M_list[4], "Actual val: ",ground_truth_M[1,1].numpy())
    print("Predicted M6 val: ", M_list[5], "Actual val: ",ground_truth_M[1,2].numpy())
    print("Predicted M7 val: ", M_list[6], "Actual val: ",ground_truth_M[2,0].numpy())
    print("Predicted M8 val: ", M_list[7], "Actual val: ",ground_truth_M[2,1].numpy())
    
#TODO: Refactor Ms. Too many parameters
def show_transform_image(rgb, M_list, ground_truth_M, should_save, current_epoch, save_every_epoch):
    plt.title("Input image")
    plt.imshow(rgb)
    if(should_save and current_epoch % save_every_epoch == 0):
        plt.savefig(gv.IMAGE_PATH_PREDICT + "/input_epoch_"+str(current_epoch)+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    f.set_size_inches(12,10)
    
    pred_M = np.ones((3,3))
    pred_M[0,0] = M_list[0]
    pred_M[0,1] = M_list[1]
    pred_M[0,2] = M_list[2]
    pred_M[1,0] = M_list[3]
    pred_M[1,1] = M_list[4]
    pred_M[1,2] = M_list[5]
    pred_M[2,0] = M_list[6]
    pred_M[2,1] = M_list[7]
    
    result = cv2.warpPerspective(rgb, ground_truth_M.numpy(), (np.shape(rgb)[1], np.shape(rgb)[0]),
                                 borderValue = (1,1,1))
    
    ax1.set_title("Ground truth")
    ax1.imshow(result)
    
    #result = cv2.perspectiveTransform(rgb, ground_truth_M.numpy())
    result = cv2.warpPerspective(rgb, pred_M, (np.shape(rgb)[1], np.shape(rgb)[0]),
                                 borderValue = (1,1,1))
    ax2.set_title("Predicted warp")
    ax2.imshow(result)
    
    if(should_save and current_epoch % save_every_epoch == 0):
        plt.savefig(gv.IMAGE_PATH_PREDICT + "/result_epoch_"+str(current_epoch)+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    print("Predicted M1 val: ", M_list[0], "Actual val: ",ground_truth_M[0,0].numpy())
    print("Predicted M2 val: ", M_list[1], "Actual val: ",ground_truth_M[0,1].numpy())
    print("Predicted M3 val: ", M_list[2], "Actual val: ",ground_truth_M[0,2].numpy())
    print("Predicted M4 val: ", M_list[3], "Actual val: ",ground_truth_M[1,0].numpy())
    print("Predicted M5 val: ", M_list[4], "Actual val: ",ground_truth_M[1,1].numpy())
    print("Predicted M6 val: ", M_list[5], "Actual val: ",ground_truth_M[1,2].numpy())
    print("Predicted M7 val: ", M_list[6], "Actual val: ",ground_truth_M[2,0].numpy())
    print("Predicted M8 val: ", M_list[7], "Actual val: ",ground_truth_M[2,1].numpy())


def visualize_M_list(M_list):
    color=iter(['r', 'g', 'b', 'y', 'c', 'm', 'y']) 
    
    for i in range(np.shape(M_list)[0]):
        c = next(color)
        visualize_individual_M(M_list[i], color = c, label = "M" + str(i))
      
    plt.show()
    
def visualize_individual_M(M0, color, label):
    x = np.random.rand(np.shape(M0)[0])

    plt.scatter(x, M0, color = color, label = label)
    plt.legend()
    plt.title("Distribution of generated M elements")
    #plt.show()

def visualize_input_data(warp_list):
    print("Input RGB distribution via norm")
    for i in range(np.shape(warp_list)[0]):
        plt.scatter(i, np.linalg.norm(warp_list[i][0,:,:]), color = 'b')
        plt.scatter(i, np.linalg.norm(warp_list[i][1,:,:]), color = 'g')
        plt.scatter(i, np.linalg.norm(warp_list[i][2,:,:]), color = 'r')
    plt.show()

def visualize_results(warp_img, rgb_img, M_list, ground_truth_M, index, p = 0.03):
    chance_to_save = np.random.rand()
    
    if(chance_to_save <= p):
        should_save = True
        least_squares_img, h = warp_perspective_least_squares(warp_img, rgb_img)
        show_transform_image_test(rgb = warp_img, least_squares_img = least_squares_img, M_list = M_list, ground_truth_M = ground_truth_M,
                                        should_save = should_save, index = index)

def visualize_blind_results(warp_img, rgb_img, M_list, index, p = 0.03):
    chance_to_save = np.random.rand()
    
    if(chance_to_save <= p):
        should_save = True
        least_squares_img, h = warp_perspective_least_squares(warp_img, rgb_img)
        show_blind_image_test(rgb = warp_img, least_squares_img = least_squares_img, 
                              M_list = M_list, ground_truth_img = rgb_img,
                              should_save = should_save, index = index)
        
def measure_ssim(warp_img, rgb_img, matrix_mean, matrix_H, matrix_own, count):
    mean_img = cv2.warpPerspective(warp_img, matrix_mean, (np.shape(warp_img)[1], np.shape(warp_img)[0]))
    h_img = cv2.warpPerspective(warp_img, matrix_H, (np.shape(warp_img)[1], np.shape(warp_img)[0]))
    own_img = cv2.warpPerspective(warp_img, matrix_own, (np.shape(warp_img)[1], np.shape(warp_img)[0]))
    rgb_img = cv2.copyMakeBorder(rgb_img, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, gv.PADDING_CONSTANT, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    SSIM = [0.0, 0.0, 0.0]; MSE = [0.0, 0.0, 0.0]; RMSE = [0.0, 0.0, 0.0]
    SSIM[0] = np.round(compare_ssim(mean_img, rgb_img, multichannel = True),4)
    SSIM[1] = np.round(compare_ssim(h_img, rgb_img, multichannel = True),4)
    SSIM[2] = np.round(compare_ssim(own_img, rgb_img, multichannel = True),4)
    
    MSE[0] = np.round(compare_mse(mean_img, rgb_img),4)
    MSE[1] = np.round(compare_mse(h_img, rgb_img),4)
    MSE[2] = np.round(compare_mse(own_img, rgb_img),4)
    
    RMSE[0] = np.round(compare_nrmse(rgb_img, mean_img),4)
    RMSE[1] = np.round(compare_nrmse(rgb_img, h_img),4)
    RMSE[2] = np.round(compare_nrmse(rgb_img, own_img),4)
    
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    f.set_size_inches(20,15)
    
    #ax1.set_title("Input image")
    ax1.imshow(warp_img)
    
    ax2.set_title("SSIM: "+str(SSIM[0])+ " MSE: "+str(MSE[0])+" RMSE: " +str(RMSE[0]))
    ax2.imshow(mean_img)
    
    ax3.set_title("SSIM: "+str(SSIM[1])+ " MSE: "+str(MSE[1])+" RMSE: " +str(RMSE[1]))
    ax3.imshow(h_img)
    
    ax4.set_title("SSIM: "+str(SSIM[2])+ " MSE: "+str(MSE[2])+" RMSE: " +str(RMSE[2]))
    ax4.imshow(own_img)
    
    #ax5.set_title("Ground truth")
    ax5.imshow(rgb_img)
    
    hide_plot_legend(ax1)
    hide_plot_legend(ax2)
    hide_plot_legend(ax3)
    hide_plot_legend(ax4)
    hide_plot_legend(ax5)
    plt.savefig(gv.IMAGE_PATH_PREDICT + "/ssim_"+str(count)+ ".png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    return SSIM, MSE, RMSE

#Visualizes a layer in the CNN
def visualize_layer(layer, resize_scale = 1):
    a,filter_range,x,y = np.shape(layer.data.numpy())
    fig = plt.figure(figsize=(y * 0.07 * resize_scale, x * 2 * resize_scale))
    #fig = plt.figure()
    
    for i in range(filter_range):
        ax = fig.add_subplot(filter_range, 3, i+1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        
        activation = layer[0,i].data.numpy()
        if(i == 0):
            print("Activation shape :", np.shape(activation))
        
        # = cv2.resize(activation, (y * resize_scale, x * resize_scale), interpolation = cv2.INTER_CUBIC)  # scale image up
        ax.imshow(np.squeeze(activation), cmap='gray')
        ax.set_title('%s' % str(i+1))
    
    plt.subplots_adjust(wspace=0, hspace=0.35)
    plt.show()

def visualize_transform_M(M_list, label, color = 'g'):
    
    X = list(range(0, np.shape(M_list)[0]))
    Y = []
    for i in range(np.shape(M_list)[0]):
        Y.append(np.linalg.norm(M_list[i]))
    
    plt.scatter(X, Y, color = color, label = label)
    plt.ylim(1.2, 3)
    plt.legend()

def visualize_predict_M(predicted_M_list):
    X = list(range(0, np.shape(predicted_M_list)[0]))
    Y = []
    for i in range(np.shape(predicted_M_list)[0]):
        #appends hard-coded 1.0s to make this a 9-vector for homography correctness
        #modified_list = np.append(predicted_M_list[i], [1.0, 1.0, 1.0], axis = 0) 
        print("Predict transform size: ", np.shape(predicted_M_list[i]))
        Y.append(np.linalg.norm(predicted_M_list[i]))
    
    plt.scatter(X, Y, color = 'r', label = "predictions")
    plt.legend()
    
    #plt.title("Distribution of norm ground-truth T vs predicted T")
    plt.show()
    
    for i in range(1, np.shape(predicted_M_list)[0]):
        diff = abs(predicted_M_list[i] - predicted_M_list[i - 1])
        plt.scatter(i - 1, np.linalg.norm(diff), color = 'b')
    
    plt.title("Differences of predicted T")
    plt.show()
    
def main():
    all_transforms = []
    predict_transforms = []
    warp_list = []
    predict_list_files = retrieve_predict_warp_list()
    for pfile in predict_list_files:
        predict_transforms.append(np.loadtxt(pfile))
    
    for batch_idx, (rgb, warp, transform) in enumerate(loader.load_dataset(batch_size = 64)):
        for t in transform:
            all_transforms.append(t.numpy())
        
        for warp_img in warp:
            warp_list.append(warp_img.numpy())
       
    visualize_transform_M(all_transforms, color = 'g', label = "training set")
    all_transforms.clear()
    all_transforms = []
        
        #if(batch_idx % 500 == 0):
            #break
        
    for batch_idx, (rgb, warp, transform) in enumerate(loader.load_test_dataset(batch_size = 64)):
        for t in transform:
            all_transforms.append(t.numpy())
        
        for warp_img in warp:
            warp_list.append(warp_img.numpy())
       
    visualize_transform_M(all_transforms, color = 'b', label = "test set")
    all_transforms.clear()
    all_transforms = []
        
        #if(batch_idx % 500 == 0):
            #break
    
    visualize_predict_M(predict_transforms)
    
    #visualize_input_data(warp_list)
    
if __name__=="__main__": #FIX for broken pipe num_workers issue.
    main()

