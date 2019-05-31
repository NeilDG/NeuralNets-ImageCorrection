# -*- coding: utf-8 -*-
"""
Code for inference and using test set
Created on Sun Mar 31 10:02:57 2019

@author: NeilDG
"""
import numpy as np
import tensorflow as tf
import cv2
from model import base_cnn
from model import fcrn_model as fcrn
from model import tensorboard_writer as tb
from matplotlib import pyplot as plt

IMAGE_H = 480;  IMAGE_W = 640
#KITTI_REDUCED_H = 128; KITTI_REDUCED_W = 416;

class InferenceCNN(object):
    
    def __init__(self,dataset):
        self.dataset = dataset
        self.batch_size = 64
        self.baseCNN = base_cnn.CNN(self.dataset)
    
    def parse_function(self, filenameRGB, fileNameDepth):
        image_string = tf.read_file(filenameRGB)
    
        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_png(image_string, channels=3)
        #image = tf.image.convert_image_dtype(image, tf.float32, saturate = True)
    
        resizedRGB = tf.image.resize_images(image, [IMAGE_H, IMAGE_W])
        
        image_string = tf.read_file(fileNameDepth)
        image = tf.image.decode_png(image_string, channels = 1)
        #image = tf.image.convert_image_dtype(image, tf.float32, saturate = True)

        resizedDepth = tf.image.resize_images(image, [IMAGE_H, IMAGE_W])
        
        return resizedRGB, resizedDepth
    
    
    def infer(self):
        trainData = self.dataset.map(map_func = self.parse_function, num_parallel_calls=4)
        batchRun = trainData.batch(self.batch_size)
        trainData = batchRun.prefetch(2)
        
        iterator = trainData.make_initializable_iterator()
        initOp = iterator.initializer
        image_rgbs, image_depths = iterator.get_next()
        
        print("Finished pre-processing train data with iterator: ", iterator)
        
        #for testing
        ground_truth = tf.placeholder(dtype = tf.float32, shape = (self.batch_size, IMAGE_H, IMAGE_W, 1), name = "ground_truth")
        testInput = tf.placeholder(dtype = tf.float32, shape = (self.batch_size, IMAGE_H, IMAGE_W, 3), name = "test_input")
        testPred = self.baseCNN.create_convNet(testInput)
        
        # Define the different metrics
        with tf.variable_scope("metrics"): 
            metrics = {"mean_squared_error" : tf.metrics.mean_squared_error(labels = ground_truth, predictions = testPred),
                       "rmse" : tf.metrics.root_mean_squared_error(labels = ground_truth, predictions = testPred)}
            tf.summary.scalar('Test/Intermediate/mean_squared_error', tf.reduce_mean(metrics["mean_squared_error"]))
            tf.summary.scalar('Test/Intermediate/rmse', tf.reduce_mean(metrics["rmse"]))
        
        # Group the update ops for the tf.metrics, so that we can run only one op to update them all
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])
        
        # Get the op to reset the local variables used in tf.metrics, for when we restart an epoch
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        metricsInitOp = tf.variables_initializer(metric_variables)
        merged = tf.summary.merge_all()
        trainWriter = tf.summary.FileWriter('train/test_result', tf.Session().graph)
        saver = tf.train.Saver()  
        globalVar = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(globalVar) #init weights, biases and other variables
            sess.run(initOp)
            sess.run(metricsInitOp)
            
            # Restore variables from disk.
            saver.restore(sess, "tmp/model_nyu_052719.ckpt") 
            
            testNum = 0;
            while True:
                  try:
                    #test image
                    inputImages = sess.run(image_rgbs)
                    depthImages = sess.run(image_depths)
                    #print("Input image shape: ", np.shape(inputImages), " Depth image shape: ", np.shape(depthImages))
                    predDepth = sess.run(testPred, feed_dict = {testInput: inputImages, ground_truth: depthImages})
                    fig = plt.imshow(inputImages[0].astype("uint8")); 
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    plt.axis("off"); plt.show()
                    fig = plt.imshow(predDepth[0][:,:,0]);  
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    plt.axis("off"); plt.show()
                    fig = plt.imshow(depthImages[0][:,:,0]);  
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    plt.axis("off"); plt.show()
                    sess.run(update_metrics_op, feed_dict = {testInput: inputImages, ground_truth: depthImages})
                    
                    # Get the values of the metrics
                    metrics_values = {k: v[0] for k, v in metrics.items()}
                    metrics_val = sess.run(metrics_values, feed_dict = {testInput: inputImages, ground_truth: depthImages})
                    print("Metrics", metrics_val)
                    
                    #summary = sess.run(merged, feed_dict = {testInput: inputImages, ground_truth: depthImages})
                    #trainWriter.add_summary(summary,testNum)
                    
                    testNum = testNum + 1
                  except tf.errors.OutOfRangeError:
                    print("End of sequence")
                    break
            
                
                