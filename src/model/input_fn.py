# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:14:51 2019

@author: NeilDG
"""

import numpy as np
import os

class InputFunction(object):
    
    #custom data dir location
    TRAIN_RGB_DIR = "C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_rgb/"
    TRAIN_DEPTH_DIR = "C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_depth/"
    
    def __init__(self):
        print("Test location ", self.TRAIN_RGB_DIR);
        
    
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
        
    def assembleRGB(self):
        print("Assembling RGB from input_fn")
        dirList = []
        folders = self.getSubDirectories(self.TRAIN_RGB_DIR)  
        for f in folders:
            SUB_DIR = "/image_02/data/"
            images = os.listdir(self.TRAIN_RGB_DIR + f + SUB_DIR)
            for i in range(len(images)):
                dirList.append(self.TRAIN_RGB_DIR + f + SUB_DIR + images[i])
            
        return dirList

    def assembleDepth(self):
        print("Assembling depth from input_fn")    
        dirList = []
        folders = self.getSubDirectories(self.TRAIN_DEPTH_DIR)  
        for f in folders:
            SUB_DIR = "/proj_depth/groundtruth/image_02/"
            images = os.listdir(self.TRAIN_DEPTH_DIR + f + SUB_DIR)
            for i in range(len(images)):
                dirList.append(self.TRAIN_DEPTH_DIR + f + SUB_DIR + images[i])
            
        return dirList
            
    def getSubDirectories(self,a_dir):
        return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
        