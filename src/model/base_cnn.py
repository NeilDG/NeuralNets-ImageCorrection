# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:36:50 2019

Own "trial-and-error" version of CNN training
@author: NeilDG
"""

import numpy as np
import tensorflow as tf
import cv2
from model import read_depth as rd
from model import conv_util
from model import fcrn_model as fcrn
from model import tensorboard_writer as tb
from matplotlib import pyplot as plt

KITTI_REDUCED_H = 128; KITTI_REDUCED_W = 416;

class CNN(object):
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 500
        self.learning_rate = 0.5
        self.batch_size = 64
        
        
    def parse_function(self, filenameRGB, fileNameDepth):
        image_string = tf.read_file(filenameRGB)
    
        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_png(image_string, channels=3)
        #image = tf.image.convert_image_dtype(image, tf.float32, saturate = True)
    
        resizedRGB = tf.image.resize_images(image, [KITTI_REDUCED_H, KITTI_REDUCED_W])
        
        image_string = tf.read_file(fileNameDepth)
        image = tf.image.decode_png(image_string, channels = 1)
        #image = tf.image.convert_image_dtype(image, tf.float32, saturate = True)

        resizedDepth = tf.image.resize_images(image, [KITTI_REDUCED_H, KITTI_REDUCED_W])
        return resizedRGB, resizedDepth
    
    def create_convNet(self, inputImage): 
        
        with tf.variable_scope("cnn", reuse = tf.AUTO_REUSE):
            conv1 = fcrn.conv(input=inputImage,name='conv1',stride=2,kernel_size=(7,7),num_filters=64)
            bn_conv1 = fcrn.batch_norm(input=conv1,name='bn_conv1',relu=True)
            pool1 = tf.nn.max_pool(bn_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool1')
            
            res2a_relu = fcrn.build_res_block(input=pool1,block_name='2a',d1=64,d2=256,projection=True,down_size=False)
            res2b_relu = fcrn.build_res_block(input=res2a_relu,block_name='2b',d1=64,d2=256)
            res2c_relu = fcrn.build_res_block(input=res2b_relu,block_name='2c',d1=64,d2=256)
            
            res3a_relu = fcrn.build_res_block(input=res2c_relu,block_name='3a',d1=128,d2=512,projection=True)
            res3b_relu = fcrn.build_res_block(input=res3a_relu,block_name='3b',d1=128,d2=512)
            res3c_relu = fcrn.build_res_block(input=res3b_relu,block_name='3c',d1=128,d2=512)
            res3d_relu = fcrn.build_res_block(input=res3c_relu,block_name='3d',d1=128,d2=512)
            
            res4a_relu = fcrn.build_res_block(input=res3d_relu,block_name='4a',d1=256,d2=1024,projection=True)
            res4b_relu = fcrn.build_res_block(input=res4a_relu,block_name='4b',d1=256,d2=1024)
            
            res5a_relu = fcrn.build_res_block(input=res4b_relu,block_name='5a',d1=512,d2=2048,projection=True)
            res5b_relu = fcrn.build_res_block(input=res5a_relu,block_name='5b',d1=512,d2=2048)
            
            layer1 = fcrn.conv(input=res5b_relu,name='layer1',stride=1,kernel_size=(1,1),num_filters=1024)
            layer1_BN = fcrn.batch_norm(input=layer1,name='layer1_BN',relu=False)
            
            # UP-CONV
            up_2x = fcrn.build_up_conv_block(input=layer1_BN,block_name='2x',num_filters=512)
            up_4x = fcrn.build_up_conv_block(input=up_2x, block_name='4x', num_filters=256)
            up_8x = fcrn.build_up_conv_block(input=up_4x, block_name='8x', num_filters=128)
            up_16x = fcrn.build_up_conv_block(input=up_8x, block_name='16x', num_filters = 64)
            #results to 128 x 416 if 2x - 4x. 256 x 832 if 2x - 4x - 8x.  512 x 1664 for 2x - 4x - 8x - 16x
            
            drop = tf.nn.dropout(up_16x, keep_prob = 1., name='drop')
            pred = fcrn.conv(input=drop,name='ConvPred',stride=1,kernel_size=(3,3),num_filters=1, trainable = True)
            pred = tf.image.resize_bicubic(pred, [KITTI_REDUCED_H, KITTI_REDUCED_W])
            print("Pred CNN shape: ", pred, "Pred type: ", type(pred))
        return pred
    
    def train(self):
        
        
        trainData = self.dataset.map(map_func = self.parse_function, num_parallel_calls=4)
        batchRun = trainData.batch(self.batch_size)
        trainData = batchRun.prefetch(2)
        
        iterator = trainData.make_initializable_iterator()
        initOp = iterator.initializer
        image_rgbs, image_depths = iterator.get_next()
        
        inputs = {"image_rgbs": image_rgbs, "image_depths": image_depths, 'iterator_init_op': initOp}
        print("Finished pre-processing train data with iterator: ", iterator)
        
#        weights = {
#            'wc1': tf.get_variable('W0', shape=(3,3,3,32), initializer=tf.contrib.layers.xavier_initializer()), 
#            'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
#            'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
#            }
#        biases = {
#            'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
#            'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
#            'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer())
#            }
    
        trainPred = self.create_convNet(inputs["image_rgbs"])
        
        ground_truths = inputs["image_depths"]
        ground_truths = tf.cast(ground_truths, tf.float32)
        loss = tf.losses.huber_loss(labels = ground_truths, predictions = trainPred)
        optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.5, use_nesterov = True).minimize(loss)
        globalVar = tf.global_variables_initializer()
        print("Successful optimizer setup")
        
        # Define the different metrics
        with tf.variable_scope("metrics"): 
            metrics = {"mean_squared_error" : tf.metrics.mean_squared_error(labels = ground_truths, predictions = trainPred),
                       "rmse" : tf.metrics.root_mean_squared_error(labels = ground_truths, predictions = trainPred)}
            tf.summary.scalar('mean_squared_error', metrics["mean_squared_error"])
            tf.summary.scalar('rmse', metrics["rmse"])
        
        # Group the update ops for the tf.metrics, so that we can run only one op to update them all
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])
        
        # Get the op to reset the local variables used in tf.metrics, for when we restart an epoch
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        metricsInitOp = tf.variables_initializer(metric_variables)
        merged = tf.summary.merge_all()
        trainWriter = tf.summary.FileWriter('/train', tf.get_default_graph())
        
        #for testing
        testInput = tf.placeholder(dtype = tf.float32, shape = (self.batch_size, KITTI_REDUCED_H, KITTI_REDUCED_W, 3), name = "test_input")
        testPred = self.create_convNet(testInput)
        
        saver = tf.train.Saver()  
#        imgNum = 0;
#        #TODO: add a batch set run and convert depth images first to inpainted versions
#        with tf.Session() as sess:
#            sess.run(globalVar)
#            sess.run(initOp)
#            
#            while True:
#                try:
#                    rgbImages = sess.run(image_rgbs)
#                    depthImages = sess.run(image_depths)
#                    print("Next batch size: " , np.shape(depthImages))
#                    k = self.batch_size
#                    for i in range(k):
#                        inpaintImg = self.inpaintDepth(depthImages[i], rgbImages[i])
#                        self.writeImg(rgbImages[i], inpaintImg, "rgb_inpaint_%s" % imgNum, "depth_inpaint_%s" % imgNum)
#                        print("Index: ", i, " K: ", k, "Img num: " ,imgNum)
#                        imgNum = imgNum + 1
#                except tf.errors.OutOfRangeError:
#                    print("Finished all batch")
#                    break
        
            
        with tf.Session() as sess:
            sess.run(globalVar) #init weights, biases and other variables
            sess.run(initOp)
            sess.run(metricsInitOp)
            tb.logGradients(sess, loss)
            # Restore variables from disk.
            saver.restore(sess, "tmp/model_0330_uint32.ckpt") 
            
            for i in range(self.epoch):
                while True:
                  try:
                    opt = sess.run([optimizer, loss])
                    tb.logGradients(sess, loss)
                    print("Optimizing! ", opt)
                    sess.run(update_metrics_op)
                  except tf.errors.OutOfRangeError:
                    print("End of sequence, looping to next epoch ", (i+1))
                    sess.run(initOp)
                    #test image
                    inputImages = sess.run(image_rgbs)
                    depthImages = sess.run(image_depths)
                    predDepth = sess.run(testPred, feed_dict = {testInput: inputImages})
                    plt.imshow(inputImages[0].astype("uint8")); plt.show()
                    plt.imshow(predDepth[0][:,:,0]); plt.show()
                    plt.imshow(depthImages[0][:,:,0]); plt.show()
                    
                    sess.run(initOp) #re-initialize iterator again for next epoch
                    
                    #save data
                    save_path = saver.save(sess, "tmp/model_0330_uint32.ckpt")
                    print("Weights saved in path: %s" %save_path)
                    break                   

                # Get the values of the metrics
                metrics_values = {k: v[0] for k, v in metrics.items()}
                metrics_val = sess.run(metrics_values)
                print("Metrics", metrics_val)
                
                summary = sess.run(merged)
                trainWriter.add_summary(summary,i)
                
        
    def inpaintDepth(self, depthImage, rgbImage):
        _,binaryInv = cv2.threshold(depthImage, 0, 1, cv2.THRESH_BINARY_INV)
        binaryInv = binaryInv.astype("uint8")
        plt.imshow(binaryInv); plt.show()
        inpaintImg = cv2.inpaint(depthImage, binaryInv, 32, cv2.INPAINT_TELEA)
        plt.imshow(inpaintImg); plt.show()
        #plt.imshow(rgbImage); plt.show()
        
        return inpaintImg
    
    def writeImg(self, rgbImg, depthImg, rgbImgName, depthImgName):
        BASE_RGB_DIR = "C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_rgb_inpaint/"
        BASE_DEPTH_DIR = "C:/Users/NeilDG/Documents/GithubProjects/NeuralNets-ImageDepthExperiment/dataset/train_depth_inpaint/"
        
        cv2.imwrite(BASE_RGB_DIR + rgbImgName + ".png", rgbImg)
        cv2.imwrite(BASE_DEPTH_DIR + depthImgName + ".png", depthImg)
        print("Image write successful", BASE_RGB_DIR, "  ", BASE_DEPTH_DIR)
                
                
                

            
        
        

