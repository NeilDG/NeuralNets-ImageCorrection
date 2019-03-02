# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:35:01 2019

@author: NeilDG
"""
import tensorflow as tf

def conv2d(x, kernel, bias, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')