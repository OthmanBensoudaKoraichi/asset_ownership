# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:56:56 2022

@author: nsuar
"""

import imageio
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

def preprocess_imagery(path):
    '''
    Pre process satellite imagery. We check for nans in the composite images,
    and then we load them into the [0,1] range.
    Input: TIF files downloaded from GEE (224x224x7)
    Output: Numpy array with shape (224x224x3). This will be the input for the ToTensor() loader.
    '''
    #reading image
    image=imageio.imread(path)[:,:,[3,2,1]] #keeping only the RGB channels (re-ordering channels so they are actually RGB)
    #fixing potential nans in image
    if np.sum(np.isnan(image))>0:
        #print(str(np.sum(np.isnan(image)))+"nans found")
        #taking the mean of the image, per channel
        mean=np.nanmean(image,axis=(0,1))
        #replacing NaN with per channel mean
        replacement=np.isnan(image)*mean
        image[np.isnan(image)]=replacement[np.isnan(image)]
    #putting image in the [0,1] range (we multiply by the satellite imagery's scaling factor)
    image=image*0.0001
    
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
    

def train_model_epoch(model,optimizer,loss_fun,data_aug,train_data,printing=False):
    '''
    trains our model for a full epoch, doing forward and backward
    propagation
    
    inputs: 
        model: a pytorch model
        optimizer: an torch.optim. optimizer
        loss_fun: an torch.nn loss function
        data_aug: a transform.Compose object with a composition of transf.
                  for data augmentation purposes
        train_data: a dataloader object with our train set

    output:
        average training loss over batches
    '''  
    #training loop
    batch_loss=0
    for X,y in train_data:
        #sending data to device
        X_batch=data_aug(X).to('cuda')
        y_batch=y.to('cuda')
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        #forward pass with gradient enabled
        
        with torch.set_grad_enabled(True):
            output = model(X_batch)
            loss = torch.sqrt(loss_fun(output, y_batch))
            #adding loss for batch (to get the mean later)
            if printing:
                print(loss.item())
            batch_loss+=loss.item()
        
        #backward pass
        loss.backward() #computes the gradient
        optimizer.step() #computes the parameter update based on the gradient
        
    
    #computing average loss over batches
    batch_loss/=len(iter(train_data))
    
    return batch_loss
    
    
    
def validation_loop(model,loss_fun,val_data,printing=False):
    '''
    computes validation (or test) error for our model. Usually to be ran after
    training each epoch of the data.
    
    inputs: 
        model: a pytorch model
        loss_fun: an torch.nn loss function
        val_data: a dataloader object with our validation set

    output:
        average validation loss over batches
    '''      
    #validation loop
    val_loss=0
    with torch.no_grad():
        for X,y in val_data:
            #sending data to device
            X_batch=X.to('cuda')
            y_batch=y.to('cuda')
                    
            pred = model(X_batch)
            loss = torch.sqrt(loss_fun(pred, y_batch))
            #adding loss for batch (to get the mean later)
            if printing:
                print(loss.item())
            val_loss+=loss.item()
        

    #computing average loss over batches
    val_loss/=len(iter(val_data))
    
    return val_loss
        
class rotation_discrete(torch.nn.Module):
    """
    rotates images with angles in a discrete set with custom probabilities.
    In this case, we don't want to rotate to anything that is not a multiple of 90 degrees.
    We can also make the images more likely to not be rotated (0 degrees rotation).
    We loop over all images in our object so we can have different angles for all of them.
    """    
    def __init__(self, angles=[0,90,180,270], probs=[0.5,0.5/3,0.5/3,0.5/3]):
        super().__init__()
        self.angles=angles
        self.probs=probs
        

    def forward(self, img):
        #numpy array with random rotations for each individual image
        angles=np.random.choice(self.angles,size=len(img), p=self.probs)
        #tensor to store rotated images
        rot_img=torch.empty(img.shape)
        for i in range(len(img)):
            #rotating individual images
            rot_img[i,:,:,:]=transforms.functional.rotate(img[i,:,:,:],angle=int(angles[i]))
        return rot_img
    
def train_and_val_model(model, optimizer, loss_fun, data_aug, train_set, train_batch_size, val_set, val_batch_size, num_epochs, best_path, train_list, val_list):
    '''
    trains a model (and performs validation) to tune hyperparameters
    
    inputs:
        model: a pytorch model
        optimizer: an torch.optim. optimizer
        loss_fun: an torch.nn loss function
        data_aug: a transform.Compose object with a composition of transf.
                  for data augmentation purposes
        train_set_small: a dataset subset to be passed to our dataloader
        train_batch_size: batch size for training dataloader
        val_set_small: a dataset subset to be passed to our dataloader
        val_batch_size: batch size for validation dataloader        
        tol: tolerance to define when our model's parameter converged
        num_epochs: number of epochs to train the model
        best_path: path to store trained model's parameters
        train_list: list to append epoch train rmse
        val_list: list to append epoch val rmse
        
    output:
        best rmse found while training.
    '''
    
    #creating dataloaders
    train_data= DataLoader( train_set ,train_batch_size, shuffle = True)
    val_data= DataLoader( val_set ,val_batch_size, shuffle = False)
    #initializing best rmse
    best_rmse=np.Inf
    
    #hyperparameter search loop
    for epoch in range(num_epochs):
        #training for one epoch
        train_rmse=train_model_epoch(model,optimizer,loss_fun,data_aug,train_data,printing=False)
        #adding rmse to list
        train_list.append(train_rmse)
        
        #validation over one epoch
        epoch_rmse=validation_loop(model,loss_fun,val_data,printing=False)
        #adding rmse to list
        val_list.append(epoch_rmse)
        
        #cleaning memory after each epoch
        torch.cuda.empty_cache()
        
        #storing parameters if the model is the best model so far:
        if epoch_rmse<best_rmse:
            best_rmse=epoch_rmse
            best_params=model.state_dict()
        
        #printing epoch and loss every 5 epochs
        #if (epoch+1)%5==0:
        print(f'Epoch {epoch+1} of {num_epochs}')
        print(f'Training average RMSE: {round(train_rmse,5)}')
        print(f'Validation average RMSE: {round(epoch_rmse,5)}')
        print('-'*25)
        
    #storing the best parameters
    torch.save(best_params, best_path)
                
    #returning best rmse
    return best_rmse

def model_frozen(model,n_unfreeze):
    '''
    takes a resnet model, freezes all conv2d parameteres except for the ones in the last n_unfreeze conv2d layers
    input:
        -pytorch model (with conv2ds modules)
        -n_unfreeze: number of conv2d layers we want to have unfrozen
    '''
    counter=0 #counter of conv2d layers
    #iterating over modules (so then we can search only conv2d modules)
    for  module in model.modules():
        #identifying conv2d modules
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            counter+=1
            #if we have a layer before the 20-n_unfreeze+1 one, we freeze it
            if counter<=20-n_unfreeze:
                for param in module.parameters():
                    param.requires_grad = False
            else: #after we found the first layer we don't want to freeze, we are done
                break
    
    return model


def experiment_store(experiment,path_results,path_train,path_val):
    '''
    stores the result of our experiments to a CSV file (via a Pandas dataframe)
    input: 
        -experiment: dictionary with the experiment
        -path_results: path to our CSV file with most of the parameters and results of the experiment
        -path_train: path to another CSV file where we store the training RMSE of each epoch of each experiment as a column
        -path_val: path to another CSV file where we store the validation RMSE of each epoch of each experiment as a column
    output:
        -there is no output
    '''
    #results of the experiment (except epoch rmse)
    try:
        #loading our dataframe to then edit it
        results1=pd.read_csv(path_results)
        
    except FileNotFoundError:
        #creating the dataset if it doesn't exists yet
        results1=pd.DataFrame(columns=['exp_number','train_batch_size','val_batch_size','data_aug','num_epochs','best_rmse','frozen'])
        
    #creating the row to add to results
    mod_exp=experiment.copy()
    del mod_exp["params_path"]
    del mod_exp["train_rmse"]
    del mod_exp["val_rmse"]
    mod_exp["data_aug"]=str(mod_exp["data_aug"])
    
    #adding row with our results
    results1=results1.append(mod_exp, ignore_index = True)
    
    #writing again pd dataframe as CSV file
    results1.to_csv(path_results,index=False)
    
    #epoch rmse results
    try:
        #loading our dataframe to then edit it
        results2=pd.read_csv(path_train)
        results3=pd.read_csv(path_val)
     
    except FileNotFoundError:
        #creating the dataset if it doesn't exists yet
        results2=pd.DataFrame()
        results3=pd.DataFrame()
     
    #adding column with our results
    results2["exp"+str(experiment["exp_number"])]=experiment["train_rmse"]
    results3["exp"+str(experiment["exp_number"])]=experiment["val_rmse"]
         
    #writing again pd dataframe as CSV file
    results2.to_csv(path_train,index=False)
    results3.to_csv(path_val,index=False)