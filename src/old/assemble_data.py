# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:48:02 2019

@author: NeilDG
"""

import tensorflow as tf
import numpy as np
import cv2
import read_depth as rd
from matplotlib import pyplot as plt
import os

class DatasetReader(object):
    
    #custom data dir location
    TRAIN_RGB_DIR = "C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_rgb/"
    TRAIN_DEPTH_DIR = "C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_depth/"
    def __init__(self):
        print("Test location ", self.TRAIN_RGB_DIR);
        
    def showSampleDepth(self):
        NEXT_DIR = "2011_09_26_drive_0001_sync\proj_depth\groundtruth\image_02/"
        #imgDir = self.TRAIN_DEPTH_DIR + NEXT_DIR + "0000000082.png";
        #testImg = rd.depth_read(imgDir)
        #plt.imshow(testImg)
        #plt.show()
        
        folders = self.getSubDirectories(self.TRAIN_DEPTH_DIR)  
        for f in folders:
            SUB_DIR = "/proj_depth/groundtruth/image_02/"
            images = os.listdir(self.TRAIN_DEPTH_DIR + f + SUB_DIR)
            for i in range(len(images)):
                img = rd.depth_read(self.TRAIN_DEPTH_DIR + f + SUB_DIR + images[i])
                plt.imshow(img)
                plt.show()
        
    
    def assembleTrainData(self):
        NEXT_DIR = "/2011_09_26_drive_0001_sync/image_02/data/"
        imgDir = self.TRAIN_RGB_DIR + NEXT_DIR + "0000000000.png"; print("Image dir: ", imgDir)
        testImg = cv2.imread(imgDir)
        plt.imshow(testImg)
        plt.show()  
        
        folders = self.getSubDirectories(self.TRAIN_RGB_DIR)  
        for f in folders:
            SUB_DIR = "/image_02/data/"
            images = os.listdir(self.TRAIN_RGB_DIR + f + SUB_DIR)
            for i in range(len(images)):
                img = cv2.imread(self.TRAIN_RGB_DIR + f + SUB_DIR + images[i])
                plt.imshow(img)
                plt.show()
    
    def getSubDirectories(self,a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]