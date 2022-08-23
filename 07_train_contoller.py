import re
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
from ENVIRONMENT import sOcnavenv
from ENVIRONMENT.sOcnavenv import SocNavEnv
from tqdm import tqdm
from UTILITY import utility 

import time
from tqdm import tqdm
import cma
# from RNN.MDN_rnn import get_responses , get_mixture_coef ,lognormal, rnn_z_loss,rnn_rew_loss,rnn_loss, MDN_RNN,MDRNNCell,flatten_parameters,unflatten_parameters,load_parameters
from RNN.RNN import LSTM,RNN

from VAE.vae import VariationalAutoencoder

time_steps = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 31
input_size = 31
vae = VariationalAutoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
vae.load_state_dict(torch.load("./MODEL/vae_model.pt"))
vae.eval()
vae.float()

latents = 31
actions = 2
hiddens = 256
gaussians = 5
epochs = 10
actions = 2

advance_split = 5
rotation_split = 5

advance_grid, rotation_grid = np.meshgrid(
    np.linspace(-1, 1, advance_split),
    np.linspace(-1, 1, rotation_split))

action_list = np.hstack((
    advance_grid.reshape(( advance_split*rotation_split, 1)),
    rotation_grid.reshape((advance_split*rotation_split, 1))))
number_of_actions = action_list.shape[0]


num_layers = 2
# # rnn = MDN_RNN(latents, actions, hiddens, gaussians).to(device)
rnn = LSTM(latents, actions, hiddens,num_layers).to(device)
# # rnn = RNN(latents, actions, hiddens).to(device)

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


def evaluate_control_model(vae, rnn, controller, device):

    env =  SocNavEnv()
    total_episodes = 20
    time_steps = 200
    s = 0
    cumulative = 0
    cumulative_ = 0
    # mdrnn = MDRNNCell(latents, actions, hiddens, gaussians).to(device)
    rnn = LSTM(latents, actions, hiddens,num_layers).to(device)
    # rnn = RNN(latents, actions, hiddens).to(device)
    rnn.load_state_dict(torch.load("./MODEL/model.pt"))
    rnn = rnn.float()

    # mdrnn = mdrnn#.double()

    while s < total_episodes:


        obs = env.reset()
        #action = torch.zeros(1, actions).to(device)
        action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        action = np.atleast_2d(action)
        action = torch.from_numpy(action).to(device)
    
        reward_ = torch.zeros(1, 1).to(device)
        hidden = [torch.zeros(1, hiddens).to(device) for _ in range(2)]
        prev_action = None
        for t in range(time_steps): 
            #env.render()
            obs = torch.from_numpy(obs)

            _, mu, log_var = vae(obs.unsqueeze(0).to(device))
            sigma = log_var.exp()
            eps = torch.randn_like(sigma)
            z = eps.mul(sigma).add_(mu)
            unsqueezed_action = action.unsqueeze(0)
            unsqueezed_z = z.unsqueeze(0)
            # mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state.items()})
            # rnn_input = torch.cat((z, action, reward_), -1).float()
            # out_full, hidden = mdrnn(rnn_input, hidden)
            rnn_input = torch.cat([unsqueezed_z, unsqueezed_action], dim=-1).float()
            _, hidden = rnn(rnn_input)
            # _,hidden,_ = rnn(rnn_input)            
            c_in = torch.cat((z, hidden[0].unsqueeze(0)),-1)
            controller.to(device)
            action_distribution = controller(c_in)
            #print(action_distribution)
            #action = action.detach().to('cpu')
            
            max_action = np.argmax(action_distribution)            
            action = action_list[max_action]
            obs, reward, done, _ = env.step(action)
            action = torch.from_numpy(action)
            reward = torch.Tensor([[reward * (1-int(done))]])
            #reward = torch.where(reward > 0 , 1, 0)
            action = action.unsqueeze(0).to(device)
            cumulative += reward
            if done:
                obs = env.reset()
                break
        
        cumulative_ += cumulative
        s+=1
    cumulative_ = cumulative / s
    return float(cumulative_)




def train_controller(controller, vae, rnn,  mode='real'):
    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                  {'popsize': 10})

    vae = vae.to(device)
    rnn = rnn.to(device)
    start_time = time.time()
    epoch = 0
    best = 0.0
    cur_best = None
    epochs = 100

    while not es.stop():
        print('epoch : {}'.format(epoch))
        solutions = es.ask()
        reward_list = []
        for s_idx, s in enumerate(solutions):
            load_parameters(s, controller)
           
            reward = evaluate_control_model(vae, rnn, controller, device)
            reward_list.append(reward)
        es.tell(solutions, reward_list)
        es.disp()

        cur_best = max(reward_list)
        best_index = np.argmax(reward_list)
        best_params = solutions[best_index]
        print('current best reward : {}'.format(cur_best))
        if not best or cur_best >= best:
            best = cur_best
            print("Saving new best with value {}...".format(cur_best))
            load_parameters(best_params, controller)

            torch.save(controller.state_dict(), './MODEL/controller.pt')

        epoch += 1
        if epoch > epochs:
            break


    es.result_pretty()


train_controller(controller, vae, rnn)
