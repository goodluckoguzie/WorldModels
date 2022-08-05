import re
import torch
from torch import nn
from torch.autograd import Variable

import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
matplotlib.style.use('ggplot')
import numpy as np
from os import listdir

batch_size = 256
input_size = 31
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Test_dataset = []
#Extract the Test_dataset 
with np.load('/dataset/Test_dataset.npz') as data:
    a = data['observations.npy']
Test_dataset = a[:6,:]

# test data
test_loader = DataLoader(Test_dataset,batch_size=batch_size,shuffle=False)

from VAE import VAE

# load your model architecture/module
model = VAE().to(device)
# fill your architecture with the trained weights
model.load_state_dict(torch.load("./saved_model/TVAE_model3.pt"))




print(test_loader)
#data = Variable(torch.from_numpy(test_loader), requires_grad=False)
for data in tqdm(test_loader):#, total=int(len(train_data)/dataloader.batch_size)):
    data = Variable(data, requires_grad=False).to(device)
    model.eval()
    with torch.no_grad():
        zs = model.get_z(data)#.data.numpy()
        zsample = model.decode(zs ).cpu()
print(zs.shape)
print("-----------------------------")
print(zsample.shape)


np.savez('dataset_',observations_input=Test_dataset,observations_output=zsample)

"""
model.eval()
with torch.no_grad():
    zsample = test_loader
    data = data.view(data.size(0), -1)
    zsample = model.get_z(data).to(device)
    #zs = model(data)#.data.numpy()
#print(zs.shape)

#print(zs.size(0))
#np.savez('zs1',zs=np.array(zs))
#np.savez('dataset_out',observations=zs)
         

def test(model, data):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        data = data.view(-1,input_size)
        recon_batch, _, _ = model(data)
        
        #for i, data in tqdm(enumerate(test_loader), total=int(len(Test_data)/dataloader.batch_size)):
            #data= data
            #data = data.to(device)
            #data = data.view(data.size(0), -1)
            #data = data.view(-1,input_size)
            #recon_batch, _, _ = model(data)
        


    
    return recon_batch
sample = Variable(torch.from_numpy(Test_data))
recon_x = test(model, sample)
print(recon_x.shape)

np.savez('Test_VAE',Test_data_orignal=np.array(Test_data),recon_x=np.array(recon_x))

"""
