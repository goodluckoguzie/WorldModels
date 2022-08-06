import numpy 
import torch
import torch.nn as nn
from RNN.RNN import RNN,Rnn
import time
import os, time, datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser("mode asigning")
parser.add_argument('--mode', type=str, required=True,help="normal,window,reward")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latents = 31
actions = 2
hiddens = 256
epochs = 20
sub_epochs = 20
train_window = 10 
batch_size = 64
timestep = 200

dataset = torch.load('./datas/saved1_rollout_rnn.pt')# our training dataset got from extract_data_for_rnn.py . note that the time step here and there must tally 


def trains(mode='normal'):
    
    if mode == 'normal':
        class MDN_Dataset(torch.utils.data.Dataset):
            def __init__(self, MDN_data):
                self.MDN_data = MDN_data
            def __len__(self):
                return len(self.MDN_data)

            def __getitem__(self, idx):
                data = self.MDN_data[idx]
                obs = data['obs_sequence']
                action = data['action_sequence']
                #reward = data['reward_sequence']
                return (action, obs)

        train_dataset = MDN_Dataset(dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        l1 = nn.L1Loss()
        rnn = RNN(latents, actions, hiddens).to(device)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
        start=time.time()
        best_loss = float("inf")
        epoch_ = []
        epoch_train_loss = []

        rnn.train()
        for epoch in range(1, epochs + 1):
            train_loss = 0
            for batch_idx, (action, obs) in enumerate(train_dataloader):# get a batch of timesteps seperated by episodes
                # we have 200 timesteps in an episode . 
                current_timestep = obs[:, :-1, :].to("cuda:0")  # remove the last timestep . input now has 199 time steps 
                action = action[:, :-1, :]          # remove the action taken in the last timestep . input now has 199 actions time steps 
                nxt_timestep = obs[:, 1:, :].to("cuda:0")  #remove the first timestep .the label contains 199 timesteps because we excluded the the fist time step
                states = torch.cat([current_timestep, action], dim=-1)
                predicted_nxt_timestep, _, _ = rnn(states)

                loss_rnn = l1(predicted_nxt_timestep, nxt_timestep)
                optimizer.zero_grad()
                loss_rnn.backward()
                train_loss += loss_rnn.item()
                optimizer.step()
            train_loss = train_loss/ len(train_dataset)


            if train_loss <= best_loss:
                if not os.path.exists('MODEL'):
                    os.makedirs('MODEL')
            
                torch.save(rnn.state_dict(), './MODEL/MDN_RNN_normal.pt')
                best_loss = train_loss
        
            epoch_.append(epoch)
            epoch_train_loss.append(train_loss)

            print('EPOCH : {} Average_loss : {}'.format(epoch, train_loss))   



    elif mode == "window":

        def create_inout_sequences(input_data,action_data, tw):
            inout_seq = []
            for i in range(timestep-tw):#the timestep is gotten from the extracted data this must tally with that of extract_data_for_rnn.py
                train_seq = input_data[:,i:i+tw,:]
                train_label = input_data[:,i+tw:i+tw+1,:]

                action_seq = action_data[:,i:i+tw,:]
                action_label = action_data[:,i+tw:i+tw+1,:]
                inout_seq.append((train_seq ,train_label,action_seq,action_label))
            return inout_seq

        class MDN_Dataset(torch.utils.data.Dataset):
            def __init__(self, MDN_data):
                self.MDN_data = MDN_data
            def __len__(self):
                return len(self.MDN_data)

            def __getitem__(self, idx):
                data = self.MDN_data[idx]
                obs = data['obs_sequence']
                action = data['action_sequence']
                #reward = data['reward_sequence']
                return (action, obs)

        train_dataset = MDN_Dataset(dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        l1 = nn.L1Loss()
        rnn = RNN(latents, actions, hiddens).to(device)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

        best_loss = float("inf")
        epoch_ = []
        epoch_train_loss = []

        rnn.train()
        for epoch in range(1, epochs + 1):
            train_loss = 0
        
            for batch_idx, (action, obs) in enumerate(train_dataloader):# get a batch of timesteps seperated by episodes
                print("batch_idx")
                print(batch_idx)

                train_inout_seq = create_inout_sequences(obs, action, train_window) #using the a sliding window of 10 . the the first 10 time step and the 11th timetep will be our label.
                                                                                    # next shift the sliding window a step ahead now our label is the 12th timestep
                for i in range(sub_epochs):# we train our model per window slides

                    w = 0
                    for current_timestep, nxt_timestep,action,_ in train_inout_seq:
        
                        action = action.to("cuda:0")  
                        current_timestep = current_timestep.to("cuda:0")  
                        nxt_timestep = nxt_timestep.to("cuda:0")  
                        states = torch.cat([current_timestep, action], dim=-1)
                        optimizer.zero_grad()
                        predicted_nxt_timestep, _, _ = rnn(states)
                        predicted_nxt_timestep = predicted_nxt_timestep[:, -1:, :] #get the last array for the predicted class

                        loss_rnn = l1(predicted_nxt_timestep, nxt_timestep)
                        optimizer.zero_grad()
                        loss_rnn.backward()
                        train_loss += loss_rnn.item()
                        optimizer.step()
                        w = w+1
                    
                    train_loss = train_loss/ len(train_dataloader)

                    if train_loss <= best_loss:
                        if not os.path.exists('model'):
                            os.makedirs('model')
                        torch.save(rnn.state_dict(), './MODEL/MDN_RNN_window.pt')
                        best_loss = train_loss
                
                    epoch_.append(sub_epochs)
                    epoch_train_loss.append(train_loss)

                    print('sub_epochs : {} Average_loss : {}'.format(i, train_loss)) 
                    print('batch_idx : {} '.format(batch_idx))
                    print('MAIN EPOCH : {} '.format(epoch)) 


    elif mode == "reward":

        def create_inout_sequences(input_data,action_data, reward_data,tw):
            inout_seq = []
            for i in range(timestep-tw):#the timestep is gotten from the extracted data this must tally with that of extract_data_for_rnn.py
                train_seq = input_data[:,i:i+tw,:]
                train_label = input_data[:,i+tw:i+tw+1,:]

                action_seq = action_data[:,i:i+tw,:]
                action_label = action_data[:,i+tw:i+tw+1,:]

                reward_seq = reward_data[:,i:i+tw,:]
                reward_label = reward_data[:,i+tw:i+tw+1,:]
                inout_seq.append((train_seq ,train_label,action_seq,action_label,reward_seq,reward_label))
            return inout_seq

        class MDN_Dataset(torch.utils.data.Dataset):
            def __init__(self, MDN_data):
                self.MDN_data = MDN_data
            def __len__(self):
                return len(self.MDN_data)

            def __getitem__(self, idx):
                data = self.MDN_data[idx]
                obs = data['obs_sequence']
                action = data['action_sequence']
                reward = data['reward_sequence']
                return (action, obs,reward)

        train_dataset = MDN_Dataset(dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        l1 = nn.L1Loss()
        rnn = Rnn(latents, actions,reward, hiddens).to(device)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

        best_loss = float("inf")
        epoch_ = []
        epoch_train_loss = []
        rnn.train()
        for epoch in range(1, epochs + 1):
            train_loss = 0
        
            for batch_idx, (action, obs,reward) in enumerate(train_dataloader):# get a batch of timesteps seperated by episodes
                print("batch_idx")
                print(batch_idx)

                train_inout_seq = create_inout_sequences(obs, action,reward, train_window) #using the a sliding window of 10 . the the first 10 time step and the 11th timetep will be our label.
                                                                        # next shift the sliding window a step ahead now our label is the 12th timestep

                for i in range(sub_epochs):# we train our model per window slides
                    w = 0
                    for current_timestep, nxt_timestep,action,_ ,reward, nxt_reward in train_inout_seq:
                        #print(reward)
                        #print(input.shape)
                        action = action.to("cuda:0")  
                        current_timestep = current_timestep.to("cuda:0")  
                        nxt_timestep = nxt_timestep.to("cuda:0")
                        reward = reward.type(torch.cuda.FloatTensor)
                        #print(reward)
                        states = torch.cat([current_timestep, action,reward], dim=-1)
                        optimizer.zero_grad()
    
                        predicted_nxt_timestep, _, _ = rnn(states)
                        predicted_nxt_timestep = predicted_nxt_timestep[:, -1:, :] #get the last array for the predicted class

                        loss_rnn = l1(predicted_nxt_timestep, nxt_timestep)
                        optimizer.zero_grad()
                        loss_rnn.backward()
                        train_loss += loss_rnn.item()
                        optimizer.step()
                        w = w+1
                    
                    train_loss = train_loss/ len(train_dataloader)

                    if train_loss <= best_loss:
                        if not os.path.exists('model'):
                            os.makedirs('model')
                        torch.save(rnn.state_dict(), './model/MDN_RNN_reward.pt')
                        best_loss = train_loss
                
                    epoch_.append(sub_epochs)
                    epoch_train_loss.append(train_loss)

                    print('sub_epochs : {} Average_loss : {}'.format(i, train_loss)) 
                    print('batch_idx : {} '.format(batch_idx))
            

                    print('MAIN EPOCH : {} '.format(epoch)) 


     
trains(args.mode)