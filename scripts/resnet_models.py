# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:59:35 2022

@author: nsuar
"""

import os
import torch
import torchvision
from torchvision import datasets



def resnet_mod():
    '''
    modifies resnet model in 2 ways:
        1-change input size in the first layer
        2-initialize first layer with initial weights for 3 RGB bands in the first layer, 
        randomly initialize the other 4 channels
    input: no input needed
    output: pytorch model

    '''
    #base model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    #retrieving initial RGB weights
    rgb_w=model.conv1.weight
    
    #modifying first layer to have input size 7 (by default, w has Xavier initialization)
    model.conv1= torch.nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    #replacing BGR channel weights (bands 2, 3 and 4) with pretrained RGB weights (reordered so they are also BGR now)
    
    with torch.no_grad(): #we don't need to differenciate this assignment, so we use no_grad
        model.conv1.weight[:,1:4,:,:]=rgb_w[:,[2,1,0],:,:]
    
    return model