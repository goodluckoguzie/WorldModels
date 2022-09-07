import torch
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import numpy 
import torch
import torch.nn as nn
import time
import os, time, datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.autograd import Variable
import numpy
from os import listdir
import os
import sys
from matplotlib.pyplot import axis
import random
import gym
from gym import spaces
import cv2
import numpy as np
import math
from ENVIRONMENT import Socnavenv
from ENVIRONMENT.Socnavenv_output import SocNavEnv
from tqdm import tqdm
from RNN.RNN import LSTM,RNN
from UTILITY import utility 
from UTILITY.utility import test_data
from UTILITY.utility import get_observation_from_dataset
from UTILITY.utility import transform_processed_observation_into_raw
import time
from tqdm import tqdm
import cma

from VAE.vae import VariationalAutoencoder


env = SocNavEnv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 62
input_size = 31
vae = VariationalAutoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
vae.load_state_dict(torch.load("./MODEL/vae_model1.pt"))
vae.eval()
vae.float()


num_layers = 2
latents = 31
actions = 2
hiddens = 256
epochs = 10
actions = 2
action_rnn = 2

advance_split = 5
rotation_split = 5

advance_grid, rotation_grid = np.meshgrid(
    np.linspace(-1, 1, advance_split),
    np.linspace(-1, 1, rotation_split))

action_list = np.hstack((
    advance_grid.reshape(( advance_split*rotation_split, 1)),
    rotation_grid.reshape((advance_split*rotation_split, 1))))
number_of_actions = action_list.shape[0]


total_episodes = 100
# num_layers = 2
# rnn = LSTM(latents, actions, hiddens,num_layers).to(device)
rnn = RNN(latents, actions, hiddens).to(device)

rnn.load_state_dict(torch.load("./MODEL/model.pt"))
rnn = rnn.float()
rnn.eval()


def flatten_parameters(params):
    return torch.cat([p.detach().view(-1) for p in params], dim=0).to('cpu').numpy()
def unflatten_parameters(params, example, device):
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def load_parameters(params, controller):
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)


class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, hiddens, actions):
        super().__init__()
        self.fc = nn.Linear(latents + hiddens, actions)

    def forward(self, inputs):
        return F.softmax(self.fc(inputs).squeeze(0).squeeze(0).detach().to('cpu'), dim=0).numpy()

controller = Controller(latents, hiddens, number_of_actions).to(device)
controller.load_state_dict(torch.load('./MODEL/controller.pt'))


def evaluate_control_model(rnn, controller, device):
   
  
    time_steps = 200
    s = 0
    cumulative = 0
    cumulative_ = 0
    
    with torch.no_grad():
        while s < total_episodes:

            obs = env.reset()
            z,_,_  = vae(torch.from_numpy(obs).unsqueeze(0).to(device))
            next_hidden = [torch.zeros(1, hiddens).to(device) for _ in range(2)]
            c_in = torch.cat((z, next_hidden[0]),-1)
            controller.to(device)
            action_distribution = controller(c_in)
            max_action = np.argmax(action_distribution)  
            action = action_list[max_action]
            nxt_obs, reward, done, _ = env.step(action)

            # for t in range(time_steps): 
            while not done:#True:
                env.render()
                nxt_z,_,_  = vae(torch.from_numpy(nxt_obs).unsqueeze(0).to(device))

                # unsqueezed_action = action.unsqueeze(0)
                unsqueezed_nxt_z= nxt_z.unsqueeze(0)
                hidden = next_hidden
                c_in = torch.cat((unsqueezed_nxt_z, hidden[0].unsqueeze(0)),-1)
                action_distribution = controller(c_in)
                max_action = np.argmax(action_distribution)  
                action = action_list[max_action]

                # action = np.atleast_2d(action)
                # action = torch.from_numpy(action).to(device)
                nxt_obs, reward, done, _ = env.step(action)

                nxt_z,_,_  = vae(torch.from_numpy(nxt_obs).unsqueeze(0).to(device))
                action = np.atleast_2d(action)
                action = torch.from_numpy(action).to(device)
                rnn_input = torch.cat([nxt_z, action], dim=-1).float()
                _, _, _, next_hidden = rnn.infer(rnn_input, hidden)
                cumulative += reward
                if done:
                    obs = env.reset()
                    break
            
            cumulative_ += cumulative
            s+=1
        cumulative_ = cumulative / s
        return float(cumulative_)


print(evaluate_control_model(rnn, controller, device))
