# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:39:48 2022

@author: nsuar
"""

import os
os.chdir("C:/Users/nsuar/Dropbox/asset-ownership/")
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.transforms import Normalize
import copy

#model=torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
#modifying base model (we modify the last fully connected layer to have output size=1 for our regression task)
model.fc=torch.nn.Linear(512, 1)

# for  module in model.modules():    
#     for param in module.parameters():
#         param.requires_grad = True


model.to('cuda')


#shallow copy of original parameters
orig_params =copy.deepcopy(model.state_dict())


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

#building dataset (images between 0 and 1, and labels (random normal))
images=Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(torch.rand(40,3,224,224))
train_set=RegData(images,torch.normal(10, 30, size=(40,1)))


#training set dataloader
train_data= DataLoader( train_set ,batch_size=10, shuffle=False,
                       generator=torch.Generator().manual_seed(0))
#optimizer
optimizer=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0)
#loss function
loss_fun = torch.nn.MSELoss()

#training loop
batch=0
for X,y in train_data:
    #sending data to device
    X_batch=X.to('cuda')
    y_batch=y.to('cuda')

        
    # zero the parameter gradients
    optimizer.zero_grad()
    
    #forward pass with gradient enabled
    with torch.set_grad_enabled(True):
        output = model(X_batch)
        loss = torch.sqrt(loss_fun(output, y_batch))
        #print(loss.item())
    
    #backward pass
    loss.backward() #computes the gradient
    optimizer.step() #computes the parameter update based on the gradient
    
    #mean of the gradient for conv1
    grad_mean=torch.mean(model.conv1.weight.grad).item()
    #max of the gradient for conv1
    grad_max=torch.max(model.conv1.weight.grad).item()
    
    print(f'Mean of gradient of conv1 layer for batch {batch}: {grad_mean}')
    print(f'Max of gradient of conv1 layer for batch {batch}: {round(grad_max,5)}')
    
    
    #comparing weights
    print("Are conv1 weights the same after update?")
    print(torch.equal(orig_params["conv1.weight"],model.state_dict()["conv1.weight"]))

    batch+=1

