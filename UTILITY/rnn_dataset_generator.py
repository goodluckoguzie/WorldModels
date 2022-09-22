import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os, time, datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
time_steps = 300


def pad_tensor( tensor, pad):
    pad_size = pad - tensor.size(0)
    return torch.cat([tensor.to(device), torch.zeros([pad_size, tensor.size(1)]).to(device)], dim=0)

# torch.from_numpy(
def fit_dataset_to_rnn(dataset):
    
    for episode_idx, episode_data in enumerate(dataset.values()):
        # print(episode_idx)

        obs_sequence = episode_data['obs_sequence']
        # print(obs_sequence.shape)
        # obs_sequence = np.asarray(obs_sequence)
        # obs_sequence = obs_sequence.tolist()
        # obs_sequence =  np.array(obs_sequence)
        # obs_sequence =  tuple(obs_sequence)
        # obs_sequence =  torch.from_numpy(obs_sequence)

        action_sequence = episode_data['action_sequence']
        reward_sequence = episode_data['reward_sequence']
        done_sequence = episode_data['done_sequence']

        obs_sequence = torch.stack(obs_sequence, dim=0).squeeze(1)

        obs_sequence = pad_tensor(obs_sequence, pad=time_steps).cpu().detach().numpy()
        # obs_sequence = utility.normalised(obs_sequence) #normilised our dataset 
        obs_sequence = torch.from_numpy(obs_sequence) 

        
        done = [int(d) for d in done_sequence]
        done = torch.tensor(done).unsqueeze(-1)
        done_sequence = pad_tensor(done, pad=time_steps)
        
        action_sequence = torch.stack(action_sequence, dim=0).squeeze(1)
        action_sequence = pad_tensor(action_sequence, pad=time_steps)

        
        reward = torch.tensor(reward_sequence).unsqueeze(-1)
        reward_sequence = pad_tensor(reward, pad=time_steps)
        reward = pad_tensor(reward, pad=time_steps)
        
        episode_data['obs_sequence'] = obs_sequence
        episode_data['action_sequence'] = action_sequence
        episode_data['done_sequence'] = done_sequence
        episode_data['reward_sequence'] = reward_sequence
    print("done padding the datset ")
    return dataset
