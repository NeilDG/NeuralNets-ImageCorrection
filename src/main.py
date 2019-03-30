# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 20:44:41 2019

@author: NeilDG
"""

from model import input_fn
from model import base_cnn


def main():
    myInput = input_fn.InputFunction()
    tfData = myInput.prepareTFData()
    
    baseCNN = base_cnn.CNN(tfData)
    baseCNN.train()
    
   
main()