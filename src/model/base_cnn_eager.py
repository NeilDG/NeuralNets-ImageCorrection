# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:25:37 2019

Base CNN model using Eager execution
@author: delgallegon
"""
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

IMAGE_H = 480;  IMAGE_W = 640
DEPTH_IMAGE_H = 240;  DEPTH_IMAGE_W = 320

class BaseCNN_Eager():
    #tf.enable_eager_execution()
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.batch_size = 64
    
    def parse_function(self, filenameRGB, fileNameDepth):
        image_string = tf.read_file(filenameRGB)
    
        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_png(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
    
        resizedRGB = tf.image.resize_images(image, [IMAGE_H, IMAGE_W])
        
        image_string = tf.read_file(fileNameDepth)
        image = tf.image.decode_png(image_string, channels = 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        #image = rd.depth_read_image(image)
        
        #resizedDepth = image
        resizedDepth = tf.image.resize_images(image, [DEPTH_IMAGE_H, DEPTH_IMAGE_W])
        return resizedRGB, resizedDepth
    
    def start_train(self):
        print("Training in eager mode? " ,tf.executing_eagerly());
        
        trainData = self.dataset.map(map_func = self.parse_function, num_parallel_calls=4)
        dataset_batch = trainData.batch(self.batch_size)
        dataset_prefetch = dataset_batch.prefetch(2)
        
        for rgb_batch, depth_batch in dataset_prefetch:
            for rgb_images in rgb_batch:
                print("Shape of RGB: " ,np.shape(rgb_images))
                print("Values: ", rgb_images.numpy())
        