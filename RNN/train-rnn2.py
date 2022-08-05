#HERE  we train our model to with window slides . for every episode we we a window of size 10 train our model to predict the next timestep
import numpy 
import torch
import torch.nn as nn
from RNN import RNN
import time
import os, time, datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

latents = 31
actions = 2
hiddens = 256
gaussians = 5
epochs = 3

timestep = 200

x = torch.load('./data/saved1_rollout_rnn.pt')# our training dataset got from extract_data_for_rnn.py . note that the time step here and there must tally 


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

train_dataset = MDN_Dataset(x)


batch_size = 256
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

l1 = nn.L1Loss()
rnn = RNN(latents, actions, hiddens).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)


train_window = 10 # 
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
        Epochs = 3
        for i in range(Epochs):# we train our model per window slides

            w = 0
            for input, labels,action,_ in train_inout_seq:
 
                action = action.to("cuda:0")  
                input = input.to("cuda:0")  
                labels = labels.to("cuda:0")  
                states = torch.cat([input, action], dim=-1)
                optimizer.zero_grad()
    
                nxt_timestep, _, _ = rnn(states)
                nxt_timestep = nxt_timestep[:, -1:, :] #get the last array for the predicted class

                loss_rnn = l1(nxt_timestep, labels)

                optimizer.zero_grad()
                loss_rnn.backward()
                train_loss += loss_rnn.item()
                optimizer.step()
                w = w+1
            
            train_loss = train_loss/ len(train_dataloader)

            if train_loss <= best_loss:
                if not os.path.exists('model'):
                    os.makedirs('model')
                torch.save(rnn.state_dict(), './model/MDN_RNN_slide.pt')
                best_loss = train_loss
        
            epoch_.append(Epochs)
            epoch_train_loss.append(train_loss)

            print('EPOCH : {} Average_loss : {}'.format(i, train_loss)) 
            print('batch_idx : {} '.format(batch_idx))
    

            print('MAIN EPOCH : {} '.format(epoch)) 

       