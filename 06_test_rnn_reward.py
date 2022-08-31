import torch
from torch import nn
import os
import sys
from matplotlib.pyplot import axis
import random
import gym
import cv2
import numpy as np
from ENVIRONMENT import Socnavenv_output
from ENVIRONMENT.Socnavenv_output import SocNavEnv
from tqdm import tqdm
from UTILITY import utility
from UTILITY.utility import transform_processed_observation_into_raw
from RNN.RNN import LSTM_reward,RNN
from torch.autograd import Variable
from UTILITY.rnn_dataset_generator import fit_dataset_to_rnn

import argparse
parser = argparse.ArgumentParser("mode asigning")
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset = torch.load('./data/saved_rnn_rollout_test.pt')
test_data = torch.load('./data/saved_vae_rollout_test.pt')
n_reward  = 1
num_layers = 2
latents = 31
actions = 2
hiddens = 256
batch_size = 1
timestep = 200
train_window = 1#0 # our sliding window value

class MDN_Dataset(torch.utils.data.Dataset):
    def __init__(self, MDN_data):
        self.MDN_data = MDN_data
    def __len__(self):
        return len(self.MDN_data)

    def __getitem__(self, idx):
        data = self.MDN_data[idx]
        obs = data['obs_sequence']
        obs = utility.normalised(obs)# normalise our observation data
        action = data['action_sequence']
        reward = data['reward_sequence']
        return (obs,action, reward)


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


dataset = fit_dataset_to_rnn(test_data)

test_dataset = MDN_Dataset(dataset)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
rnn = LSTM_reward(latents, actions,n_reward, hiddens,num_layers).to(device)
# rnn = RNN(latents, actions, hiddens).to(device)
rnn.load_state_dict(torch.load("./MODEL/model_n.pt"))




def code_and_decode(model, data):
    data = torch.from_numpy(data)
    data = Variable(data, requires_grad=False).to(device)
    with torch.no_grad():
        output,_ = model(data)
        output = output.cpu()
        
    return output 

    
rnn.eval()
for batch_idx, (obs,action, reward) in enumerate(test_dataloader):# get a batch of timesteps seperated by episodes

# for batch_idx, (action, obs) in enumerate(train_dataloader):

    # print(batch_idx)
    # print(obs.shape)

    train_inout_seq = create_inout_sequences(obs, action,reward, train_window)#get the action data in batches along with the expected true value
    for current_timestep, nxt_timestep,action,_,reward, nxt_reward in train_inout_seq:

        action = action.to(device) 
        current_timestep = current_timestep.to(device) 
        nxt_timestep = nxt_timestep.to(device)
        reward = reward.type(torch.cuda.FloatTensor)

        states = torch.cat([current_timestep, action,reward], dim=-1)

        predicted_nxt_timestep, _ = rnn(states)
        # predicted_nxt_timestep, _,_ = rnn(states)

        current_timestep = current_timestep
        nxt_timestep = nxt_timestep
        predicted_nxt_timestep = predicted_nxt_timestep[:, -1:, :]


        steps = predicted_nxt_timestep.shape[0]

        for step in range(steps):
            current_timestep = current_timestep[0, train_window-1, :]
            current_timestep = current_timestep.cpu().detach().numpy()

            nxt_timestep = nxt_timestep[0, step, :]
            nxt_timestep = nxt_timestep.cpu().detach().numpy()

            predicted_nxt_timestep = predicted_nxt_timestep[0, step, :]
            predicted_nxt_timestep = predicted_nxt_timestep.cpu().detach().numpy()


            print("output_sample")
            print(predicted_nxt_timestep)
            print("input_sample")
            print(current_timestep)
            print("")

            current_timestep = np.atleast_2d(current_timestep)
            current_timestep = utility.denormalised(current_timestep)
            current_timestep = current_timestep.flatten()

            nxt_timestep = np.atleast_2d(nxt_timestep)
            nxt_timestep = utility.denormalised(nxt_timestep)
            nxt_timestep = nxt_timestep.flatten()

            predicted_nxt_timestep = np.atleast_2d(predicted_nxt_timestep)
            predicted_nxt_timestep = utility.denormalised(predicted_nxt_timestep)
            predicted_nxt_timestep = predicted_nxt_timestep.flatten()

            print("output_sample")
            print(predicted_nxt_timestep)
            print("input_sample")
            print(current_timestep)
            print("")
            # Check if all elements in array are zero. this help us to remove the padded timestep
            result = np.all((current_timestep == 0))
            if result:
                print('Array contains only 0')
                continue
            else:
                print('Array has non-zero items too')
            robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(current_timestep)
            image0 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "current timestep", dont_draw=True)
            current_grey = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            cv2.imshow("current_timestep", image0)

            robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(nxt_timestep)
            image1 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "next timestep", dont_draw=True)
            next_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            cv2.imshow("next timestep", image1)

            robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(predicted_nxt_timestep)
            image2 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "predicted next timestep", dont_draw=True)
            next_predicted_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            cv2.imshow("predicted next timestep", image2)

            merged = cv2.merge([current_grey, next_grey, next_predicted_grey])
            # print(merged.shape, merged.dtype)
            #merged = np.dstack((current_grey, next_grey, next_predicted_grey))
            print(merged.shape, merged.dtype)
            cv2.imshow("Merged", merged)


            for j in tqdm(range(100)):
                k = cv2.waitKey(10)
                if k%255 == 27:
                    sys.exit(0)