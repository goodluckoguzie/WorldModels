import numpy 
import torch
import torch.nn as nn
from RNN.RNN import LSTM,RNN
import time
import os, time, datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
from UTILITY.early_stopping_for_rnn import  EarlyStopping
from UTILITY import utility
from UTILITY.rnn_dataset_generator import fit_dataset_to_rnn

parser = argparse.ArgumentParser("mode asigning")
parser.add_argument('--epochs', type=int, help="epochs")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latents = 31
actions = 2
hiddens = 256
epochs = args.epochs
batch_size = 2048
timestep = 200
num_layers = 2

# train_dataset = torch.load('./data/saved_rnn_rollout_train.pt')# our training dataset got from extract_data_for_rnn.py . note that the time step here and there must tally 
# val_dataset = torch.load('./data/saved_rnn_rollout_validation.pt')# our training dataset got from extract_data_for_rnn.py . note that the time step here and there must tally 



train_data = torch.load('./data/saved_vae_rollout_train.pt')
val_data = torch.load('./data/saved_vae_rollout_validation.pt')
train_dat = fit_dataset_to_rnn(train_data)
val_dat = fit_dataset_to_rnn(val_data)



    
class MDN_Dataset(torch.utils.data.Dataset):
    def __init__(self, MDN_data):
        self.MDN_data = MDN_data
    def __len__(self):
        return len(self.MDN_data)

    def __getitem__(self, idx):
        data = self.MDN_data[idx]
        obs = data['obs_sequence']
        # obs = utility.normalised(obs)# normalise our observation data
        action = data['action_sequence']
        #reward = data['reward_sequence']
        return (action, obs)

train_dataset = MDN_Dataset(train_dat)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#load our training dataset 

val_dataset = MDN_Dataset(val_dat)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#load our validation dataset 
l1 = nn.MSELoss()

rnn = LSTM(latents, actions, hiddens,num_layers).to(device)
# rnn.load_state_dict(torch.load("./MODEL/model1.pt"))

optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)



def train_model(model, batch_size, patience, n_epochs):
    
    # track our training loss as the model trains
    train_losses = []
    # track our validation loss as the model trains
    valid_losses = []
    # track our average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # eval_losses = []
    # losses_rnn = []
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() #activate model for training
        for batch_idx, (action, obs) in enumerate(train_dataloader):# get a batch of timesteps seperated by episodes

            # we have 200 timesteps in an episode . 
            current_timestep = obs[:, :-1, :].to(device)  # remove the last timestep . input now has 199 time steps 
            action = action[:, :-1, :]          # remove the action taken in the last timestep . input now has 199 actions time steps

            nxt_timestep = obs[:, 1:, :].to(device)  #remove the first timestep .the label contains 199 timesteps because we excluded the the fist time step
            states = torch.cat([current_timestep, action], dim=-1) 
            predicted_nxt_timestep, _= rnn(states)           

            # calculate the loss
            loss_rnn = l1(predicted_nxt_timestep, nxt_timestep)
            loss_rnn.backward()
            optimizer.step()
            train_losses.append(loss_rnn.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # activate our model for evaluation
        for batch_idx, (action, obs) in enumerate(train_dataloader):# get a batch of timesteps seperated by episodes

            # we have 200 timesteps in an episode . 
            current_timestep = obs[:, :-1, :].to(device)  # remove the last timestep . input now has 199 time steps 
            action = action[:, :-1, :]          # remove the action taken in the last timestep . input now has 199 actions time steps

            nxt_timestep = obs[:, 1:, :].to(device)  #remove the first timestep .the label contains 199 timesteps because we excluded the the fist time step
            states = torch.cat([current_timestep, action], dim=-1) 
            predicted_nxt_timestep, _= rnn(states)     

            # calculate the loss
            val_loss_rnn = l1(predicted_nxt_timestep, nxt_timestep)
            valid_losses.append(val_loss_rnn.item())  



        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.sum(train_losses)/len(train_dataset)
        valid_loss = np.sum(valid_losses)/len(train_dataset)
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
        if epoch % 5 == 0:
            early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./MODEL/model1.pt'))

    return  model, avg_train_losses, avg_valid_losses




# early stopping patience; how long to wait after last time validation loss improved.
patience = 100

model, train_loss, valid_loss = train_model(rnn, batch_size, patience, epochs)

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
#plt.show(



