import sys
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
import argparse
import matplotlib.pyplot as plt
import utility

parser = argparse.ArgumentParser("epochs asigning")
parser.add_argument('--epochs', type=int, help="epochs")
parser.add_argument('--max_samples', type=int, help="max_samples", nargs='?', const=0, default=0)
args = parser.parse_args()

import VAE

# check vae dir exists, if not, create it
vae_dir = 'savedmodel'
if not os.path.exists(vae_dir):
    os.makedirs(vae_dir)

DIR_NAME = 'outputs'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

# leanring parameters
epochs = args.epochs
batch_size = 32
input_size = 31
z_dim = 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Val_dataset = []
Train_dataset = []
Test_dataset = []

# Extract the train dataset 
with np.load('./data/Train_dataset.npz') as data:
    train_data = data['observations.npy']
train_data = utility.normalised(train_data)

# Filter some samples to improve speed while debugging
if args.max_samples > 0:
    train_data = train_data[:args.max_samples,:]
print(f'max_samples: {args.max_samples}')
print(f'dataset shape: {train_data.shape}')


#Extract the validation dataset 
with np.load('./data/Val_dataset.npz') as data:
    val_data = data['observations.npy']
val_data = utility.normalised(val_data)

#Extract the Test_dataset 
with np.load('./data/Test_dataset.npz') as data:
    test_data = data['observations.npy']
    print("ddddddddddddddddddddddddddddddddddddddddddd")
    print(test_data.shape)
test_data = utility.normalised(test_data)

# for i in range(31):
#     print(i, np.min(train_data[:, i]), np.max(train_data[:, i]))
# sys.exit(0)

# training and validation data loaders
train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
val_loader   = DataLoader(val_data,   batch_size=batch_size,shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=batch_size,shuffle=False)


model = VAE(input_size=input_size, z_dim=z_dim, hidden_dim=200).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def validate(model, dataloader):
    model.eval()
    running_loss = 0.0

    bs = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            data = data.view(data.size(0), -1)
            bs += len(data)
            recon_batch, mu, logvar = model(data)
            loss = model.loss(recon_batch, data, mu, logvar)
            running_loss += loss.item()
        print("Validation Loss: {:.5f}   (per output: {:.5f})".format(running_loss/bs, running_loss/bs/input_size))

    val_loss = running_loss/len(dataloader)
    return val_loss




train_loss = []
val_loss = []
for epoch in range(args.epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    
    model.train()
    running_loss = 0.0
    for data in tqdm(train_loader):#, total=int(len(train_data)/dataloader.batch_size)):
    # for data in train_loader:#, total=int(len(train_data)/dataloader.batch_size)):
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    bs = len(train_loader)
    print("Epoch[{}/{}] Loss: {:.5f}".format(epoch+1, epochs, running_loss/bs))

    #print('====> Epoch: {} done!'.format(epoch))
    train_epoch_loss = running_loss/len(train_loader)
    val_epoch_loss = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

    torch.save(model.state_dict(), './saved_model/TVAE_model.pt')

#epochs = range(1,35)
#plt.plot(epochs, train_epoch_loss, 'g', label='Training loss')
#plt.plot(epochs, val_epoch_loss, 'b', label='validation loss')
#plt.title('Training and Validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

    

"""

#Extract the Test_dataset 
with np.load('Test_dataset.npz') as data:
    a = data['observations.npy']
x = np.array(a)
for i in range(1):#len(1)):#(x)):
    Test_dataset.append(x[i]) 
Test_dataset = np.array(Test_dataset)
# test data
Test_data = Test_dataset

data = Variable(torch.from_numpy(Test_data), requires_grad=False)
model.eval()
zs = model.get_z(data).data.numpy()
np.savez('zs',zs=np.array(zs))


#data = Variable(torch.from_numpy(x), requires_grad=False)
#model.train()
#zs = model.get_z(data).data.numpy()
#print(zs.shape)
#s = model.decode(torch.from_numpy(zs))
#print(s.shape)
    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)])
        data = Variable(data, requires_grad=False)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss(recon_batch, data, mu, logvar)
        loss.backward()
        #train_loss += loss.data[0]
        optimizer.step()
        if (i % log_interval == 0) and (epoch % 5 ==0):
            #Print progress
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size*batches_per_epoch,
                #loss.data[0] / len(data)))
                loss.data / len(data)))

    print('====> Epoch: {} done!'.format(epoch))
"""