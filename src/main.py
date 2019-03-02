# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:44:41 2019

@author: NeilDG
"""

import tensorflow as tf
import numpy as np
from model import input_fn

def main():
    myInput = input_fn.InputFunction()
    myData = myInput.assembleTrainingData();
    
    dataset = tf.data.Dataset.from_tensor_slices((myData[0],myData[1]))
    print(dataset)
    
main()