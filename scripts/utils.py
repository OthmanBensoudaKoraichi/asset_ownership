# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:56:56 2022

@author: nsuar
"""

import imageio
import numpy as np
import torch
from torchvision.transforms import Normalize

def preprocess_imagery(path):
    '''
    Pre process satellite imagery. We check for nans in the composite images,
    convert RGB bands from 16-bit to 8-bit,
    
    Input: TIF files downloaded from GEE (224x224x7)
    Output: Numpy array with shape (224x224x7). This will be the input for the ToTensor() loader.
    '''
    #reading image
    image=imageio.v3.imread(path)
    #fixing potential nans in image
    if np.sum(np.isnan(image))>0:
        #print(str(np.sum(np.isnan(image)))+"nans found")
        #taking the mean of the image, per channel
        mean=np.nanmean(image,axis=(0,1))
        #replacing NaN with per channel mean
        replacement=np.isnan(image)*mean
        image[np.isnan(image)]=replacement[np.isnan(image)]
    #converting RGB bands to 8-bits
    image[:,:,1:4]=(image[:,:,1:4]*0.0255).astype('uint8')
    
    return image


#defining class for our dataset 
class RegData(torch.utils.data.Dataset):
    '''
    inputs: 
        images: Tensor with dimensions (number of images x channels x height x width)
        labels: Tensor with dimensions (number of images x 1)
    '''
    def __init__(self, images,labels):
        self.images=images
        self.labels=labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    

def normalize_custom(tensor,param=True):
    '''
    Normalizes our data, but replacing mean and SD for RGB channel with the
    correct values used for ResNet
    input: 
        -tensor: tensor object with our imagery with dimensions (N,7,H,W)
        -param: dummy to return mean and SD or not
    outputs:
        -list of 3 objects (if param==True). The first object contains our Tensor normalized,
        the second one contains the means of the 7 channels and the third one contains
        their standard deviations
        -normalized tensor if param==False
    '''
    #mean and SD
    mean=torch.mean(tensor, dim = [0,2,3])
    sd=torch.std(tensor, dim = [0,2,3])
    
    #replacing BGR channel means and SD (bands 2, 3 and 4) with pretrained RGB means and SD (reordered so they are also BGR now)
    mean[1:4]=torch.Tensor([0.406,0.456,0.485])
    sd[1:4]=torch.Tensor([0.225, 0.224,0.229])
    
    #normalizing our Tensor
    tensor=Normalize(mean=mean,std=sd)(tensor)
    
    if param==True:
        return [tensor,mean,sd]
    else:
        return tensor
   
    
def train_model(model,optimizer,loss_fun,dataloader):
    '''
    trains our model for a full epoch, doing forward and backward
    propagation
    
    inputs: 
        model: a pytorch model
        optimizer: an torch.optim. optimizer
        loss_fun: an torch.nn loss function
        dataloader: an dataloader with our dataset

    output:
        nothing
        prints the loss for each batch?
    
    '''  
    for X,y in iter(dataloader):
        #sending data to device
        X_batch=X.to('cuda')
        y_batch=y.to('cuda')
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        #forward pass with gradient enabled
        with torch.set_grad_enabled(True):
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            print(loss)
        
        #backward pass
        loss.backward() #computes the gradient
        optimizer.step() #computes the parameter update based on the gradient