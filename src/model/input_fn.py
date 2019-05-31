# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:14:51 2019

@author: NeilDG
"""

import numpy as np
import tensorflow as tf
import os

class InputFunction(object):
    
    #custom data dir location
    TRAIN_RGB_DIR = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_rgb/"
    TRAIN_DEPTH_DIR = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_depth/"
    
    TEST_RGB_DIR = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/val_rgb/"
    TEST_DEPTH_DIR = "D:/Users/delgallegon/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/val_depth/"
    
    def __init__(self):
        print("Started assembly of input data using TF")
        
    
    def assembleTrainingData(self):
        rgbList = []; depthList = []
        rgbFolders= self.getSubDirectories(self.TRAIN_RGB_DIR);
        depthFolders = self.getSubDirectories(self.TRAIN_DEPTH_DIR);
        
        for f in depthFolders:
            DEPTH_SUB_DIR = "/proj_depth/groundtruth/image_02/"
            RGB_SUB_DIR = "/image_02/data/"
            images = os.listdir(self.TRAIN_DEPTH_DIR + f + DEPTH_SUB_DIR)
            for i in range(len(images)):
                rgbImagePath = self.TRAIN_RGB_DIR + f + RGB_SUB_DIR + images[i]
                depthImagePath = self.TRAIN_DEPTH_DIR + f + DEPTH_SUB_DIR + images[i]
                if(os.path.exists(rgbImagePath)):
                    depthList.append(depthImagePath)
                    rgbList.append(rgbImagePath)
        
        return rgbList, depthList
    
    def assembleTestData(self):
        rgbList = []; depthList = []
        rgbFolders= self.getSubDirectories(self.TEST_RGB_DIR);
        depthFolders = self.getSubDirectories(self.TEST_DEPTH_DIR);
        
        for f in depthFolders:
            DEPTH_SUB_DIR = "/proj_depth/groundtruth/image_02/"
            RGB_SUB_DIR = "/image_02/data/"
            images = os.listdir(self.TEST_DEPTH_DIR + f + DEPTH_SUB_DIR)
            for i in range(len(images)):
                rgbImagePath = self.TEST_RGB_DIR + f + RGB_SUB_DIR + images[i]
                depthImagePath = self.TEST_DEPTH_DIR + f + DEPTH_SUB_DIR + images[i]
                if(os.path.exists(rgbImagePath)):
                    depthList.append(depthImagePath)
                    rgbList.append(rgbImagePath)
        
        return rgbList, depthList
    
    def prepareTFData(self):
        myData = self.assembleTrainingData()
        dataset = tf.data.Dataset.from_tensor_slices((myData[0],myData[1]))
        #dataset = dataset.shuffle(len(myData))
        return dataset;
    
    def prepareTFTestData(self):
        myData = self.assembleTestData()
        dataset = tf.data.Dataset.from_tensor_slices((myData[0],myData[1]))
        #dataset = dataset.shuffle(len(myData))
        #print("Data RGB shape: ", np.shape(myData[0]), " Data depth shape: ", np.shape(myData[1]))
        return dataset;
    
         
    def getSubDirectories(self,a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
        