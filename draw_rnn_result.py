import torch
from torch import nn
import os
import sys
from matplotlib.pyplot import axis
import random
import gym
import cv2
import numpy as np
from ENVIRONMENT import socnavenv
from ENVIRONMENT.socnavenv import SocNavEnv
from tqdm import tqdm
from UTILITY import utility
from UTILITY.utility import transform_processed_observation_into_raw
from RNN import RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_dataset = torch.load('./data/saved1_rollout_rnn.pt')

latents = 31
actions = 2
hiddens = 256


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
        return  (action, obs)



train_dataset = MDN_Dataset(train_dataset)





batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

rnn = RNN(latents, actions, hiddens).to(device)

rnn.load_state_dict(torch.load("./MODEL/MDN_RNN_normal.pt"))



rnn.eval()
for batch_idx, (action, obs) in enumerate(train_dataloader):

  
    input = obs[:, :-1, :].to("cuda:0") # remove the last timestep
    action = action[:, :-1, :]  # remove the last action and timestep
    Nxt_states = obs[:, 1:, :].to("cuda:0")  #remove the first timestep and include the last timestep

    states = torch.cat([input, action], dim=-1)
    predicted_Nxt_states, _, _ = rnn(states)

    input_sample = Nxt_states
    output_sample = predicted_Nxt_states

    # = utility.denormalised(input_sample)
    #print(output_sample)

    steps = output_sample.shape[0]
  
    for step in range(steps):
      
        input_sample = input_sample[0, step, :]
        input_sample = input_sample.cpu().detach().numpy()

        output_sample = output_sample[0, step, :]
        output_sample = output_sample.cpu().detach().numpy()

        # Check if all elements in array are zero
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
