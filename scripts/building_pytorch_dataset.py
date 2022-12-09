import os
os.chdir("C:/Users/nsuar/Dropbox/asset-ownership/")
from scripts.utils import preprocess_imagery, RegData
import pandas as pd
import torch
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import random_split


#loading data
dataset=pd.read_csv("data/processed/ground_truth.csv")
dataset=dataset[dataset["year"] > 2013]

images=[] #list to store individual tensors for each image
labels=torch.empty(0) #empty tensor to store labels

for i in dataset.index:
#loading sample of 100 images
#for i in range(100):
    try:
        images.append(ToTensor()(preprocess_imagery('data/imagery/image'+str(i)+'.tif')).float())
        labels=torch.cat((labels,torch.FloatTensor([dataset['hv271'][i]])))
    except FileNotFoundError:
        pass 

#normalizing the data (and converting them from list to tensor)
images=torch.stack(images,0)
images=Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(images)

#building dataset (converting list of images to tensor, adding extra dimension to labels)
regdataset=RegData(images,labels.unsqueeze(1))

#splitting into train, validation and test sets
train_set,val_set,test_set= random_split(regdataset, [0.8, 0.1, 0.1],
                                         generator=torch.Generator().manual_seed(1910))

# train_set_small,val_set_small,test_set_small= random_split(regdataset, [0.8, 0.1, 0.1],
#                                          generator=torch.Generator().manual_seed(1910))


torch.save(train_set, "data/pytorch/train_set.pt")
torch.save(val_set, "data/pytorch/val_set.pt")
torch.save(test_set, "data/pytorch/test_set.pt")

# torch.save(train_set_small, "data/pytorch/train_set_small.pt")
# torch.save(val_set_small, "data/pytorch/val_set_small.pt")
# torch.save(test_set_small, "data/pytorch/test_set_small.pt")