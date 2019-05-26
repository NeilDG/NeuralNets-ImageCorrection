# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:35:01 2019

@author: NeilDG
"""
import tensorflow as tf

#Creates a binary mask given ground-truth depth data
def create_binary_mask(depth_data):
    binary_mask = tf.where(tf.equal(depth_data, 0), tf.zeros_like(depth_data), tf.ones_like(depth_data))
    return binary_mask

#Produces a feature image based from available depth data. 
#A feature image only considers pixel values where depth information is also available.
def reduce_features(rgb_image, depth_data):
    binary_mask = create_binary_mask(depth_data)
    features = tf.multiply(rgb_image,binary_mask)
    return features
