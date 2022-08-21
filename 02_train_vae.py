import sys
import torch
import torchvision
import torch.optim as optim
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt
from UTILITY import utility
import os, time, datetime
rng = np.random.default_rng()
from UTILITY.early_stopping_for_vae import  EarlyStopping

parser = argparse.ArgumentParser("epochs asigning")
parser.add_argument('--epochs', type=int, help="epochs")
parser.add_argument('--max_samples', type=int, help="max_samples", nargs='?', const=0, default=0)
args = parser.parse_args()

from VAE.vae import VariationalAutoencoder
# check vae dir exists, if not, create it
vae_dir = 'MODEL'
if not os.path.exists(vae_dir):
    os.makedirs(vae_dir)


# leanring parameters
epochs = args.epochs
batch_size = 64
input_size = 31
z_dim = 31
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Val_dataset = []
Train_dataset = []

train_dataset = torch.load('./data/saved_vae_rollout_train.pt')
val_dataset = torch.load('./data/saved_vae_rollout_validation.pt')


class VAE_Dataset(torch.utils.data.Dataset):
    def __init__(self, obs_data):
        self.obs_data = obs_data
        
    def __len__(self):
        return len(obs_data)

    def __getitem__(self, idx):
        data = obs_data[idx]
        return data


def flating_obs_data(data):
    imgs = []
    for episode_data in data.values():
        
        imgs = imgs + episode_data['obs_sequence']
    print('obs_sequence: {}'.format(len(imgs)))
    imgs = torch.stack(imgs, dim=0)
    print('obs_dataset.size :', imgs.size())
    return imgs

obs_data = flating_obs_data(train_dataset)
obs_data = utility.normalised(obs_data)
train_obs_data = VAE_Dataset(obs_data)

val_obs_data = flating_obs_data(val_dataset)
val_obs_data = utility.normalised(val_obs_data)
val_obs_data = VAE_Dataset(obs_data)

# train_obs_data = VAE_Dataset( utility.normalised(flating_obs_data(train_dataset)))
# val_obs_data = VAE_Dataset(utility.normalised(flating_obs_data(val_dataset)))


if args.max_samples > 0:
    train_obs_data = train_obs_data[:args.max_samples,:]
    val_obs_data = val_obs_data[:args.max_samples,:]

print(f'max_samples: {args.max_samples}')


# training and validation data loaders
train_loader = DataLoader(train_obs_data, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_obs_data,   batch_size=batch_size, shuffle=False)


# model = Autoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
model= VariationalAutoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, batch_size, patience, n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        #for data in tqdm(train_loader):
        for batch, data in enumerate(train_loader, 1):
            data = data.to(device)
            #torch.from_numpy(data)
            data = data.view(-1, input_size)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output,_,_  = model(data)
            # calculate the loss
            # loss = ((data - output)**2).mean()
            loss = ((data - output)**2).sum()
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################with torch.no_grad():
        model.eval() # prep model for evaluation
        for data in val_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            data = data.to(device)
            #torch.from_numpy(data)
            data = data.view(data.size(0), -1)
            output,_,_ = model(data)
            # calculate the loss
            loss = ((data - output)**2).sum()
            # loss = ((data - output)**2).mean()
            # record validation loss
            valid_losses.append(loss.item())
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./MODEL/vae_model.pt'))

    return  model, avg_train_losses, avg_valid_losses


# early stopping patience; how long to wait after last time validation loss improved.
patience = 250

model, train_loss, valid_loss = train_model(model, batch_size, patience, epochs)

# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')


#epochs = range(1,35)
#plt.plot(epochs, train_epoch_loss, 'g', label='Training loss')
#plt.plot(epochs, val_epoch_loss, 'b', label='validation loss')
#plt.title('Training and Validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()
