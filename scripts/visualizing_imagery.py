import os
import numpy as np
import pandas as pd
from geetools import cloud_mask
import requests, zipfile, io
os.chdir("C:/Users/nsuar/Dropbox/asset-ownership/data/")

#importing Earth Engine packages
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate()  #every person needs an Earth Engine account to do this part
ee.Initialize()

dataset=pd.read_csv("processed/ground_truth.csv")
dataset['original_index'] = dataset.index

#temporarily keeping only data with Landsat 8 imagery
dataset=dataset[dataset["year"] > 2013]


#defining image
startDate = '2021-01-01'
endDate = '2021-12-31'
landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
# filter date
landsat = landsat.filterDate(startDate, endDate) 
#applying cloud masking
landsat_masked=landsat.map( cloud_mask.landsatSR(['cloud']) )
#selecting bands
landsat_masked=landsat_masked.select(["B2","B3","B4"])
landsat = landsat.select(["B2","B3","B4"])

#defining parameters for the function
image_res=30
n_pixels=224

#visualization parameters
visParams={'min': 0, 'max': 3000, 'gamma': 1.4,  
           'bands' : ['B4', 'B3', 'B2'], 'dimensions' : str(n_pixels)+"x"+str(n_pixels),
           'format' : 'jpg'}

#defining the function
def visualization(point,name,mask=True):
    '''
    Function to visualize the images for our ML application.
    Inputs:
        -point= ee.Geometry.point object
        -name: name that is going to be given to the jpg file
        -mask: True to get masked image, False to get unmasked image
    Outputs:
        The function doesn't produce an output, but generates a file called
        "name.jpg" in the current directory
    '''
    #computing bounding box
    len=image_res*n_pixels # for landsat, 30 meters * 224 pixels
    region= point.buffer(len/2).bounds().getInfo()['coordinates']
    coords=np.array(region)
    coords=[np.min(coords[:,:,0]), np.min(coords[:,:,1]), np.max(coords[:,:,0]), np.max(coords[:,:,1])]
    rectangle=ee.Geometry.Rectangle(coords)
    
    #clipping the area from satellite image
    if mask==True:
        clipped_image= landsat_masked.mean().clip(rectangle)
    else:
        clipped_image= landsat.mean().clip(rectangle)
        
    #getting the image
    requests.get(clipped_image.getThumbUrl(visParams))
    open(name+'.jpg', 'wb').write(requests.get(clipped_image.getThumbUrl(visParams)).content)
    
 
os.chdir("C:/Users/nsuar/Dropbox/asset-ownership/data/example_rgb_images/")
#looping over points (Angola, Chad, Egypt and Kenya)
for i in [264,18376,6030,10550]:
    visualization(point=ee.Geometry.Point(dataset['LONGNUM'][i],dataset['LATNUM'][i] ),
                  name='image'+str(i))
    