import numpy as np
import pandas as pd
import pickle
import time
import math  
from geetools import cloud_mask
    
def image_task(image,point,image_res,n_pixels,folder_name, image_name,storage="Cloud"):
    
    """
    function to download satellite images from a ee.imageCollection object.
    We first generate a bounding box of image_res*n_pixels meters around "point",
    then we clip that region from the image collection, take the mean image from the collection,
    and send that as a task to the Google Earth Engine. 
    After that, we download the image Google Cloud Storage if storage=="Cloud", 
    or to Google Drive if storage=="Drive".
    
    Inputs:
    -image= ee.ImageCollection object
    -point= ee.Geometry.Point object
    -image_res= resolution of the image in meters
    -n_pixels= number of pixels to extract on the images
    -storage= string indicating if we are storing the images in Google Cloud or Google Drive.
              Defaults to Google Cloud.
    -folder_name= string with Google Cloud bucket name if storage=="Cloud"
                  string with the name of a folder in the root of Google Drive if storage=="Drive"
    -image_name= string with the image_name for the TIFF image.

    Output:
     task= an EE task object. we can then use task.status() to check the status of the task.
     If the task is completed, we will see a TIFF image in "folder_name" with name "image_name.tif".
     The image has 3 dimensions, where the first 2 are n_pixels, and the 3rd is the number of bands of "image".
    """
    #generating the box around the point
    len=image_res*n_pixels # for landsat, 30 meters * 224 pixels
    region= point.buffer(len/2).bounds().getInfo()['coordinates']
    #defining the rectangle
    coords=np.array(region)
    #taking min and maxs of coordinates to define the rectangle
    coords=[np.min(coords[:,:,0]), np.min(coords[:,:,1]), np.max(coords[:,:,0]), np.max(coords[:,:,1])]
    rectangle=ee.Geometry.Rectangle(coords)

    #generating the export task ( dimensions is "WIDTHxHEIGHT"  )
    if storage=="Cloud":
        task=ee.batch.Export.image.toCloudStorage(image=image.filterBounds(rectangle).mean(), 
                            bucket=folder_name, 
                            description=image_name, 
                            region=str(region), dimensions=str(n_pixels)+"x"+str(n_pixels))
    if storage=="Drive":
        task=ee.batch.Export.image.toDrive(image=image.filterBounds(rectangle).mean(), 
                            folder=folder_name, 
                            description=image_name, 
                            region=str(region), dimensions=str(n_pixels)+"x"+str(n_pixels))
    
    
    #starting the task
    task.start()
    return task

def imagery(year,masked=True):
    ''' function to decide what imagery we are using depending on the year, and
    to take composites from the relevant year for our ground truth '''
    
    if year>2013: #in this case we use landsat 8       
        #getting collection
        landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")
        
        #filtering dates
        landsat=landsat.filterDate(str(year)+'-01-01', str(year)+'-12-31') 
        
        #applying cloud masking
        landsat_masked=landsat.map( cloud_mask.landsatSR(['cloud']) )
        
        # selecting bands for both versions
        landsat_masked=landsat_masked.select(["B1","B2","B3","B4","B5","B6","B7"]) 
        landsat = landsat.select(["B1","B2","B3","B4","B5","B6","B7"]) 
            
    else:
        pass
        
    # filtering date (taking a full year)
    if masked==True:
        return landsat_masked
    else:
        return landsat


#importing Earth Engine packages
import ee #install in the console with "pip install earthengine-api --upgrade"
ee.Authenticate() #every person needs an Earth Engine account to do this part
ee.Initialize()


#reading dataset
dataset=pd.read_csv("ground_truth.csv")
dataset['original_index'] = dataset.index

# column for task status
dataset['task_status']='_'

#Keeping only data with Landsat 8 imagery
dataset=dataset[dataset["year"] > 2013]

#list to store the task objects
tasks=[]

#running the tasks to get the imagery
batch_size=100
starting_index=0

for j in range( math.ceil(N/batch_size) ):
    #determining batch lower and upper indexes, given batch size
    #lower is always fixed
    lower_i=starting_index+batch_size*j
    #upper can vary at the end of the list
    if batch_size*(j+1)>N:
        upper_i=starting_index+N
    else:
        upper_i=starting_index+batch_size*(j+1)
        
        
    #generating the tasks for all the images in the batch
    for i in range(lower_i,upper_i):
        tasks.append(image_task(image=imagery(dataset['year'][i],masked=True),
                                point=ee.Geometry.Point(dataset['LONGNUM'][i],dataset['LATNUM'][i] ),
                                image_res=30,
                                n_pixels=224,
                                folder_name='imagery',
                                image_name='image_'+dataset.attrs['name']+str(i),
                                storage="Drive"))
        
    #printing message:
    print('Batch '+str(j+1)+': Retrieving images '+str(lower_i+1)+' to '+str(upper_i)+' of a total of '+str(starting_index+N))
      
    #checking status of the mentioned tasks
    batch_status=dataset.loc[lower_i:upper_i-1,'task_status'].value_counts() #counting status of the tasks
    
    while batch_status.get('COMPLETED',0) + batch_status.get('FAILED',0)< upper_i - lower_i: #checking that not all tasks are done
        time.sleep(10) #running the code every 10 seconds
        for i in range(lower_i,upper_i):
            #checking status of each task
            if dataset.loc[i,'task_status']=='_' or dataset.loc[i,'task_status']=='READY' or dataset.loc[i,'task_status']=='RUNNING':
                dataset.loc[i,'task_status']=tasks[i-starting_index].status()['state'] #use when restarting list of tasks
                #dataset.loc[i,'task_status']=tasks[i].status()['state']
                                    
        #updating batch status
        batch_status=dataset.loc[lower_i:upper_i-1,'task_status'].value_counts()
        #reporting them back
        print('Status of batch '+str(j+1)+':')
        print('completed images= '+str(batch_status.get('COMPLETED',0)))
        print('failed images= '+str(batch_status.get('FAILED',0)))
        print('pending images= '+str(upper_i-lower_i -batch_status.get('COMPLETED',0)-batch_status.get('FAILED',0)))
        print('------------------')
        
    #updating dataset after every batch
    dataset.to_csv(dataset.attrs['name']+'.csv',index=False)

print('The Landsat download code has finished')