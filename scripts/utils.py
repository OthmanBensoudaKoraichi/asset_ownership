# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:56:56 2022

@author: nsuar
"""

import imageio
import numpy as np


def preprocess_imagery(image):
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



#dataloader (fix code)
from torchvision.transforms import ToTensor
image = ToTensor()(image).unsqueeze(0).float()

