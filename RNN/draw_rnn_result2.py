
import torch
import torch.nn as nn
from RNN import RNN

import os, time, datetime,sys
from tqdm import tqdm
import random
import gym
import cv2
import numpy as np
import socnavenv
from socnavenv import SocNavEnv
from tqdm import tqdm
from utility import transform_processed_observation_into_raw


device = 'cuda' if torch.cuda.is_available() else 'cpu'

latents = 31
actions = 2
hiddens = 256

timestep = 200 # must be the same as extract_data_for_rnn.py 

x = torch.load('./data/saved1_rollout_rnn.pt')



def create_inout_sequences(input_data,action_data, tw):
    inout_seq = []
    for i in range(timestep-tw):#the timestep is gotten from the extracted data
        train_seq = input_data[:,i:i+tw,:] #get the 1 to 10 , label is the 11th - get 2 to 11 label 12th .....
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


batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

rnn = RNN(latents, actions, hiddens).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
rnn.load_state_dict(torch.load("./model/MDN_RNN_slide.pt"))

train_window = 10 # our sliding window value

rnn.eval()

for batch_idx, (action, obs) in enumerate(train_dataloader):
    print(batch_idx)
    print(obs.shape)

    train_inout_seq = create_inout_sequences(obs, action, train_window) #get the action data in batches along with the expected true value

    for input, labels,action,_ in train_inout_seq:
 
        action = action.to("cuda:0")  
        input = input.to("cuda:0")  
        labels = labels.to("cuda:0")  
        states = torch.cat([input, action], dim=-1)

        predicted_Nxt_states, _, _ = rnn(states)
        input_sample = labels
        output_sample = predicted_Nxt_states[:, -1:, :]

        # = utility.denormalised(input_sample)
        #print(output_sample)

        steps = output_sample.shape[0]
  
        for step in range(steps):
        
            input_sample = input_sample[0, step, :]
            input_sample = input_sample.cpu().detach().numpy()

            output_sample = output_sample[0, step, :]
            output_sample = output_sample.cpu().detach().numpy()


            print("output_sample")
            print(output_sample)
            print("input_sample")
            print(input_sample)
            print("")
            # Check if all elements in array are zero. this help us to remove the padded timestep
            result = np.all((input_sample == 0))
            if result:
                print('Array contains only 0')
                continue
            else:
                print('Array has non-zero items too')

            input_sample = np.atleast_2d(input_sample)
            #input_sample = utility.denormalised(input_sample)
            input_sample = input_sample.flatten()

            output_sample = np.atleast_2d(output_sample)
            #utility.denormalised(output_sample)
            output_sample = output_sample.flatten()

            print("output_sample")
            print(output_sample)
            print("input_sample")
            print(input_sample)
            print("")


            robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(input_sample)
            image1 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, " next timestep", dont_draw=True)
            cv2.imshow(" next timestep", image1)
            (B, _, R) = cv2.split(image1)

            robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(output_sample)
            image2 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "predicted next timestep", dont_draw=True)
            (_, G, _) = cv2.split(image2)
            cv2.imshow("predicted next timestep", image2)

            merged = cv2.merge([B, G,R])
            cv2.imshow("Merged", merged)


            for j in tqdm(range(100)):
                k = cv2.waitKey(10)
                if k%255 == 27:
                    sys.exit(0)

