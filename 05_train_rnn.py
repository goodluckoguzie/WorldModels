import numpy 
import torch
import torch.nn as nn
# from RNN.RNN import LSTM,RNN
import time
import os, time, datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
from UTILITY.early_stopping_for_rnn import  EarlyStopping_1 as EarlyStopping
from UTILITY import utility
from UTILITY.rnn_dataset_generator import fit_dataset_to_rnn
import torch
import torch.nn as nn
import numpy as np
import yaml
import argparse

import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# train_window = 1 
# timestep = 300



train_data = torch.load('./Data/saved_vae_rollout_train.pt')
val_data = torch.load('./Data/saved_vae_rollout_validation.pt')
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


def create_inout_sequences(input_data,action_data, tw):
    inout_seq = []
    for i in range(Agent.timestep-tw):#the timestep is gotten from the extracted data this must tally with that of extract_data_for_rnn.py
        train_seq = input_data[:,i:i+tw,:]
        train_label = input_data[:,i+tw:i+tw+1,:]

        action_seq = action_data[:,i:i+tw,:]
        action_label = action_data[:,i+tw:i+tw+1,:]
        inout_seq.append((train_seq ,train_label,action_seq,action_label))
    return inout_seq


class LSTM(nn.Module):

    def __init__(self, n_latents,n_actions, n_hiddens, n_layers):            
        # super().__init__()
        super(LSTM, self).__init__()
        self.n_latents = n_latents
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        

        self.lstm = nn.LSTM(input_size=n_latents+n_actions,hidden_size=n_hiddens, num_layers=n_layers,batch_first=True,dropout=0.65)

        # HE-Initialisierung
        weight = torch.zeros(n_layers,n_hiddens)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)


        self.classifier = nn.Linear(n_hiddens, n_latents)

    def init_hidden(self):      
 
        hidden_state = torch.zeros(self.lstm.num_layers,Agent.batch_size,self.lstm.hidden_size)
        cell_state = torch.zeros(self.lstm.num_layers,Agent.batch_size,self.lstm.hidden_size)
        return (hidden_state, cell_state)

    def forward(self, x):
        self.hidden = self.init_hidden()
        # _, (hidden, _) = self.lstm(x)                  
        h,h_out = self.lstm(x)                  

        # out=hidden[-1]                                  
        # return self.classifier(out)
        y = self.classifier(h)
        return y,None,h_out


class RNN_LSTM():
    def __init__(self, config:str, **kwargs) -> None:
        assert(config is not None)
        # initializing the env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # rnn variables
        self.n_latents = None
        self.n_actions = None
        self.n_hiddens = None
        self.num_layers = None

        # self.buffer_size = None
        self.num_episodes = None
        # self.epsilon = None
        # self.epsilon_decay_rate = None
        self.batch_size = None
        # self.gamma = None
        # self.lr = None
        self.timestep = None
        self.train_window = None
        self.save_path = None
        # self.render_freq = None
        self.save_freq = None
        self.run_name = None
        # print("dddddddddddddddddddddd",self.run_name )

                # setting values from config file
        self.configure(self.config)


        # declaring the network
        self.RNN = LSTM(self.n_latents, self.n_actions, self.n_hiddens,self.num_layers).to(self.device)
        # print(self.RNN)
        # print("yes)")
        
    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.n_latents is None:
            self.n_latents = config["n_latents"]
            assert(self.n_latents is not None), f"Argument n_latents size cannot be None"

        if self.n_hiddens is None:
            self.n_hiddens = config["n_hiddens"]
            assert(self.n_hiddens is not None), f"Argument hidden_layers cannot be None"

        if self.n_actions is None:
            self.n_actions = config["n_actions"]
            assert(self.n_actions is not None), f"Argument n_actions cannot be None"

        if self.num_layers is None:
            self.num_layers = config["num_layers"]
            assert(self.num_layers is not None), f"Argument num_layers cannot be None"

        if self.num_episodes is None:
            self.num_episodes = config["num_episodes"]
            assert(self.num_episodes is not None), f"Argument num_episodes cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), f"Argument batch_size cannot be None"

        if self.save_path is None:
            self.save_path = config["save_path"]
            assert(self.save_path is not None), f"Argument save_path cannot be None"

        if self.run_name is None:
            self.run_name = config["run_name"]
            assert(self.run_name is not None), f"Argument save_path cannot be None"

        if self.train_window is None:
            self.train_window = config["train_window"]
            assert(self.train_window is not None), f"Argument save_path cannot be None"


        if self.timestep is None:
            self.timestep = config["timestep"]
            assert(self.timestep is not None), f"Argument save_path cannot be None"


        if self.save_freq is None:
            self.save_freq = config["save_freq"]
            assert(self.save_freq is not None), f"Argument save_freq cannot be None"



        # variable to keep count of the number of steps that has occured
        self.steps = 0
        # check vae dir exists, if not, create it
        RNN_runs = 'RNN_runs'
        if not os.path.exists(RNN_runs):
            os.makedirs(RNN_runs)
        if self.run_name is not None:
            self.writer = SummaryWriter('RNN_runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

        self.train_dataset = MDN_Dataset(train_dat)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)#load our training dataset 

        val_dataset = MDN_Dataset(val_dat)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)#load our validation dataset 
        self.l1 = nn.MSELoss()

    def plot(self, episode):
        self.Train_loss.append(self.train_loss)
        self.Valid_loss.append(self.valid_loss)


        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "Train_loss"), np.array(self.train_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "Valid_loss"), np.array(self.valid_loss), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("Train_loss / epoch", self.train_loss, episode)
        self.writer.add_scalar("valid_loss / epoch", self.valid_loss, episode)


    def train_model(self):
        self.Train_loss = []
        self.Valid_loss = []
            # early stopping patience; how long to wait after last time validation loss improved.
        patience = 100
        
        # track our training loss as the model trains
        self.train_losses = []
        # track our validation loss as the model trains
        self.valid_losses = []
        # track our average training loss per epoch as the model trains
        self.avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        self.avg_valid_losses = [] 
        
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        # eval_losses = []
        # losses_rnn = []
        print("dddddd",self.num_episodes)
        for epoch in range(0, self.num_episodes):

            ###################
            # train the model #
            ###################
            # model.train() #activate model for training
            rnn = self.RNN
            optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

            rnn.train() #activate model for training

            for batch_idx, (action, obs) in enumerate(self.train_dataloader):# get a batch of timesteps seperated by episodes
                # print("batch_idx")
                # print(batch_idx)

                train_inout_seq = create_inout_sequences(obs, action, Agent.train_window) #using the a sliding window of 10 . the the first 10 time step and the 11th timetep will be our label.
                # w = 0                                   
                                # next shift the sliding window a step ahead now our label is the 12th timestep
                for current_timestep, nxt_timestep,action,_ in train_inout_seq:
                    
                    
                    # we have 200 timesteps in an episode . 
                    action = action.to(device)
                    current_timestep = current_timestep.to(device)
                    # print("curent ",current_timestep.shape)
                    optimizer.zero_grad()  
                    nxt_timestep = nxt_timestep.to(device)
                    states = torch.cat([current_timestep, action], dim=-1) 
                    predicted_nxt_timestep, _ ,_= rnn(states)
                    predicted_nxt_timestep = predicted_nxt_timestep[:, -1:, :] #get the last array for the predicted class
                    # calculate the loss
                    loss_rnn = self.l1(predicted_nxt_timestep, nxt_timestep)
                    loss_rnn.backward()
                    optimizer.step()
                    self.train_losses.append(loss_rnn.item())

            ######################    
            # validate the model #
            ######################
            rnn.eval() # activate our model for evaluation
            for batch_idx, (action, obs) in enumerate(self.val_dataloader):# get a batch of timesteps seperated by episodes
                # print("batch_idx")
                # print(batch_idx)

                train_inout_seq = create_inout_sequences(obs, action, Agent.train_window) #using the a sliding window of 10 . the the first 10 time step and the 11th timetep will be our label.
                # w = 0                                                                   # next shift the sliding window a step ahead now our label is the 12th timestep
                for current_timestep, nxt_timestep,action,_ in train_inout_seq:

                    # we have 200 timesteps in an episode . 
                    action = action.to(device)
                    current_timestep = current_timestep.to(device) 
                    nxt_timestep = nxt_timestep.to(device)
                    states = torch.cat([current_timestep, action], dim=-1) 
                    # forward pass: compute predicted outputs by passing inputs to the model
                    predicted_nxt_timestep, _,_= rnn(states)
                    predicted_nxt_timestep = predicted_nxt_timestep[:, -1:, :] #get the last array for the predicted class
                    # calculate the loss
                    val_loss_rnn = self.l1(predicted_nxt_timestep, nxt_timestep)
                    self.valid_losses.append(val_loss_rnn.item())  
                    # w = w+1



            # print training/validation statistics 
            # calculate average loss over an epoch
            self.train_loss = np.sum(self.train_losses)/(len(self.train_dataset))
            self.valid_loss = np.sum(self.valid_losses)/(len(self.val_dataloader))
            self.avg_train_losses.append(self.train_loss)
            self.avg_valid_losses.append(self.valid_loss)

            self.plot(epoch+1)

            # saving model
            if (self.save_path is not None) and ((epoch+1)%self.save_freq == 0):
                if not os.path.isdir(self.save_path):
                    os.makedirs(self.save_path)
                try:
                    self.save_model(os.path.join(self.save_path, "episode"+ str(epoch+1).zfill(8) + ".pth"))
                except:
                    print("Error in saving model")
                
            epoch_len = len(str(self.num_episodes))
            
            print_msg = (f'[{epoch:>{epoch_len}}/{self.num_episodes :>{epoch_len}}] ' +
                        f'train_loss: {self.train_loss:.5f} ' +
                        f'valid_loss: {self.valid_loss:.5f}')
            
            print(print_msg)
            
            # clear lists to track next epoch
            self.train_losses = []
            self.valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            if epoch % 10 == 0:
                early_stopping(self.valid_loss, rnn)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # load the last checkpoint with the best model
        rnn.load_state_dict(torch.load('./MODEL/rnn_model_layer_1.pt'))

        return  rnn, self.avg_train_losses, self.avg_valid_losses


# config file for the model
config = "./configs/RNN_hidden_256_layer_1.yaml"
    # declaring the network
Agent = RNN_LSTM(config, run_name="RNN_hidden_256_layer_1")


# print(config)
rnn, train_loss, valid_loss = Agent.train_model()


    
