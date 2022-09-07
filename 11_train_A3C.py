#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import math
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, sys
import gym
import gym.envs.box2d
import cv2

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from agents import A3C
from torchvision import transforms
from collections import deque
from os.path import join, exists
# from models import *

from collections import namedtuple
from RNN.RNN import LSTM,RNN

from VAE.vae import VariationalAutoencoder

time_steps = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
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
gaussians = 5
epochs = 1
actions = 2

# num_layers = 2
DEVICE = 'cpu'
rnn = RNN(latents, actions, hiddens).to(DEVICE)
# rnn = LSTM(latents, actions, hiddens,num_layers).to(device)

from ENVIRONMENT.Socnavenv import SocNavEnv



Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


# device = 'cuda' if torch.cuda.is_available() else 'cpu'

logdir = 'logs'

# MAX_R = 1.

transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


# def obs2tensor(obs):
#     # binary_road = obs2feature(obs) # (10, 10)
#     binary_road = obs # (10, 10)

#     s = binary_road.flatten()
#     s = torch.tensor(s.reshape([1, -1]), dtype=torch.float)
#     obs = np.ascontiguousarray(obs)
#     # obs = torch.tensor(obs, dtype=torch.float)
#     obs = transform(obs).unsqueeze(0)
#     return obs.to(device), s.to(device)


# def obs2feature(s):
#     upper_field = s[:84, 6:90] # we crop side of screen as they carry little information
#     img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
#     upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
#     upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation = cv2.INTER_NEAREST) # re scaled to 7x7 pixels
#     upper_field_bw = upper_field_bw.astype(np.float32)/255
#     return upper_field_bw


def set_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# def train_process(global_agent, vae, rnn, update_term, pid, state_dims, hidden_dims, lr, device=None, seed=0):
def train_process(global_agent, vae, rnn, update_term, state_dims, hidden_dims, lr, device=None, seed=0):
    set_seed(seed)
    env =  SocNavEnv()

    agent = A3C(input_dims=state_dims, hidden_dims=hidden_dims, lr=lr)
    agent.load_state_dict(global_agent.state_dict())

    scores = []
    running_means = []
    step = 0
    for ep in range(1_000_000):
        obs = env.reset()
        score = 0.
        i = 0
        next_hidden = [torch.zeros(1, 1, 256).to(device) for _ in range(2)]
        for _ in range(1):
            # env.render()
            next_obs, reward, done, _ = env.step(agent.possible_actions[-2])
            score += reward

        while not done:#True:

            hidden = next_hidden          
            state = torch.cat([torch.from_numpy(next_obs).unsqueeze(0), hidden[0].squeeze(0)], dim=1)           
            action, p = agent.select_action(state) # nparray, tensor
            next_obs, reward, done, _ = env.step(action.reshape([-1]))

            with torch.no_grad():
                action = torch.tensor(action, dtype=torch.float).view(1, -1).to(device)
                rnn_input = torch.cat([torch.from_numpy(next_obs).unsqueeze(0), action], dim=-1) #
                rnn_input = rnn_input.view(1, 1, -1)
                # rnn.infer().to (DEVICE)
                _, _, _, next_hidden = rnn.infer(rnn_input, hidden)


            next_state = torch.cat([torch.from_numpy(next_obs).unsqueeze(0), next_hidden[0].squeeze(0)], dim=1)

            # Scores
            score += reward
            
            if done:
                
                # print("done")

                reward_tensor = torch.tensor([reward], dtype=torch.float).to(device)
                agent.replay.push(state.data, p, reward_tensor, next_state.data)
    
                
                # running_mean = np.mean(scores[-30:])
                running_mean = np.mean(scores)

                print('Ep: {}, Replays: {}, Running Mean: {:.2f}, Score: {:.2f}' .format( ep, len(agent.replay), running_mean, score))
                scores.append(score)

                running_means.append(running_mean)
        
                optim = torch.optim.Adam(global_agent.parameters(), lr=lr)
                optim.zero_grad()
                agent.update(done)
                for g_param, param in zip(global_agent.parameters(), agent.parameters()):
                    g_param._grad = param.grad
                optim.step()
                agent.load_state_dict(global_agent.state_dict())
                
                break

            else:
                reward_tensor = torch.tensor([reward], dtype=torch.float).to(device)
                agent.replay.push(state.data, p, reward_tensor, next_state.data)
            
            if len(agent.replay) == update_term:
                optim = torch.optim.Adam(global_agent.parameters(), lr=lr)
                optim.zero_grad()
                agent.update(done)
                for g_param, param in zip(global_agent.parameters(), agent.parameters()):
                    g_param._grad = param.grad
                optim.step()
                agent.load_state_dict(global_agent.state_dict())

            i += 1
            step += 1
#         agent.update()
   
        pdict = {
            'agent': agent,
            'scores': scores,
            'avgs': running_means,
            'step': step,
            'n_episodes': ep,
            'seed': seed,
            'update_term': update_term,
        }
        worse = 0
        best_score = 0
        if score > -1.5:
            best_agent_state = global_agent.state_dict()
            save_ckpt(pdict, 'A3C({:03d})-{}.pth.tar'.format(int(score), ep))
            worse = -20
        elif score > best_score:
            best_score = score
            best_agent_state = global_agent.state_dict()
            save_ckpt(pdict, 'A3C({:03d})-{}.pth.tar'.format(int(score), ep))
            worse = -20
        else:
            worse += 1
            if worse > -20 and best_agent_state is not None:
                    global_agent.load_state_dict(best_agent_state)
    env.close()
    save_ckpt(pdict, 'A3C({:03d})-{}.pth.tar'.format(int(score), ep))
    return pdict


def save_ckpt(info, filename, root='ckpt', add_prefix=None, save_model=True):
    if add_prefix is None:
        ckpt_dir = os.path.join(root, type(info['agent']).__name__)
    else:
        ckpt_dir = os.path.join(root, add_prefix, type(info['agent']).__name__)
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if save_model:
        torch.save(
            info, '{}/{}.pth.tar'.format(ckpt_dir, filename)
        )
    # plt.figure()
    # plt.plot(info['scores'])
    # plt.plot(info['avgs'])
    # plt.savefig('{}/scores-{}.png'.format(ckpt_dir, filename))

def save_means_plot(infos, add_prefix=None, root='ckpt'):
    if add_prefix is None:
        ckpt_dir = os.path.join(root, type(infos[0]['agent']).__name__)
    else:
        ckpt_dir = os.path.join(root, add_prefix, type(infos[0]['agent']).__name__)
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    plt.figure()
    for info in infos:
        plt.plot(info['avgs'])
    plt.savefig('{}/total-scores.png'.format(ckpt_dir))


# ### V model & M model

# vae_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*.pth.tar')))[-1]
# vae_state = torch.load(vae_path, map_location={'cuda:0': str(device)})

# rnn_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
# rnn_state = torch.load(rnn_path, map_location={'cuda:0': str(device)})

# vae = VAE(hp.vsize).to(device)
# vae.load_state_dict(vae_state['model'])
# vae.eval()


rnn.load_state_dict(torch.load("./MODEL/model.pt"))
rnn = rnn.float()
rnn.eval()

# rnn = MDNRNN(hp.vsize, hp.asize, hp.rnn_hunits, hp.n_gaussians).to(device)
# rnn = RNN(hp.vsize, hp.asize, hp.rnn_hunits).to(device)
# rnn.load_state_dict(rnn_state['model'])
# mdnrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})


# print('Loaded VAE: {}, RNN: {}'.format(vae_path, rnn_path))

# ###  Environment

total_infos = []
max_ep = 200
test_ep = 200

state_dims = 287
hidden_dims = 200
lr = 1e-4

global_agent = A3C(input_dims=state_dims, hidden_dims=hidden_dims, lr=lr).cpu()
global_agent.share_memory()

update_term = 100
n_processes = 0
processes = []


train_process(global_agent, vae, rnn, update_term, state_dims, hidden_dims, lr,)

# for pid in range(n_processes+1):
#     # if pid == 0:
#     #     p = mp.Process(target=test_process, args=(global_agent, vae, rnn, update_term, pid, state_dims, hidden_dims, lr,))
#     # else:
#     p = mp.Process(target=train_process, args=(global_agent, vae, rnn, update_term, pid, state_dims, hidden_dims, lr,))
#     p.start()
#     processes.append(p)

# for p in processes:
#     p.join()

