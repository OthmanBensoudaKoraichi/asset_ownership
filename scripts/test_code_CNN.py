# # -*- coding: utf-8 -*-
# """
# Created on Thu Nov 17 01:10:42 2022

# @author: nsuar
# """
# import os
# import torch
# import torchvision
# from torchvision import datasets


# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# baia=model.conv1.weight.cpu().detach().numpy()

# model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# #modifying first layer
# model2.conv1= torch.nn.Conv2d(7, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# baia2=model2.conv1.weight.cpu().detach().numpy()

# import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

# # sample execution (requires torchvision)
# from PIL import Image
# from torchvision import transforms
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

# with torch.no_grad():
#     output = model(input_batch)
    
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)



# os.chdir("C:/Users/nsuar/Dropbox/asset-ownership/data/")

# image_datasets=datasets.ImageFolder('temp')

# batch=torch.utils.data.DataLoader(image_datasets, batch_size=1)

# input = next(iter(batch))

# from PIL import Image

# imagen = Image.open('temp\clase\image_nicolas19.tif')

# imagen = Image.open('imagery/image_nicolas21.tif')

# import imageio
# imagen=imageio.v3.imread('imagery\image_nicolas21.tif')
# image7 = ToTensor()(imagen).unsqueeze(0).float()


# import numpy as np


# #making rgb bands 8-bit
# image = (imagen/256).astype('uint8')

# from torchvision.transforms import ToTensor

# #test image with 7 bands
# image7 = ToTensor()(imagen).unsqueeze(0).float()

# #test image with only 3 bands
# image3= ToTensor()(imagen[:,:,0:3]).unsqueeze(0).float()


# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = image7.to('cuda')
#     model.to('cuda')
    



# with torch.no_grad():
#     output = model(input_batch)
    
    
    
    
    
### newer code (using dataloader and stuff)

import os
os.chdir("C:/Users/nsuar/Dropbox/asset-ownership/")
from scripts.utils import preprocess_imagery, RegData, normalize_custom
from scripts.resnet_models import resnet_mod
import pandas as pd
import torch
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data.dataloader import DataLoader


#loading base model
model=resnet_mod()

#modifying base model (resnet outputs 1000x1, we pass that through relu, and then linear to output 1 score)
model=torch.nn.Sequential(model,
                          torch.nn.ReLU(),
                          torch.nn.Linear(1000, 1)).to('cuda')

print(model)

#loading data
dataset=pd.read_csv("data/processed/ground_truth.csv")
dataset=dataset[dataset["year"] > 2013]

images=[] #list to store individual tensors for each image
labels=torch.empty(0) #empty tensor to store labels

#for i in dataset.index:
#loading sample of 100 images
for i in range(100):
    try:
        images.append(ToTensor()(preprocess_imagery('data/imagery/image'+str(i)+'.tif')).float())
        labels=torch.cat((labels,torch.FloatTensor([dataset['hv271'][i]])))
    except FileNotFoundError:
        pass 

#normalizing the data (and converting them from list to tensor)
images=torch.stack(images,0)
images=normalize_custom(images,param=False)

#building dataset (converting list of images to tensor, adding extra dimension to labels)
regdataset=RegData(images,labels.unsqueeze(1))

#example of our dataloader
train_data= DataLoader( regdataset ,10, shuffle = True)


optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.MSELoss()

num_epochs=50
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)    
    train_model(model,optimizer,loss_fn,train_data)
    

# for epoch in range(num_epochs):
#     print(f'Epoch {epoch}/{num_epochs - 1}')
#     print('-' * 10)
    
# #just checking the size of each batch, and how to use iter and next
# baia=DataLoader( torch.arange(10) ,1, shuffle = True)

# for x in iter(baia):
#     print(x)
    
    


# for batch in iter(train_data):
#     print(batch.size())

# print(next(iter(train_data))[1].size())



# #sending model and first batch to cuda
# if torch.cuda.is_available():
#     input_batch = next(iter(train_data))[0].to('cuda')
#     model.to('cuda')
    
# #doing one forward pass
# with torch.no_grad():
#     output = model(input_batch)

# #checking that output is 10x1
# print(output)


# #computing loss
# (output,next(iter(train_data))[1].to('cuda'))
# print(loss)

# # pending: torch.save, 
# # check resnet example, extract their parameters for normalization

# optimizer (definition based on model)