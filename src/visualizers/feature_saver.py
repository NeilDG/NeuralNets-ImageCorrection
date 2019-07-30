# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:02:26 2019

@author: delgallegon
"""
import torch

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()