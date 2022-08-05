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
#from Socnavenv import SocNavEnv
#from sOcnavenv import SocNavEnv
#from draw_socnavenv import SocNavEnv
from tqdm import tqdm
from rnn import Rnn, RNN,LSTM


from UTILITY import utility 
from utility import test_data
from utility import get_observation_from_dataset
from utility import transform_processed_observation_into_raw
import time
from tqdm import tqdm
import cma
import utility
env = SocNavEnv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


latents = 31
actions = 2
hiddens = 256
reward = 1
#rnn = Rnn(latents, actions,reward, hiddens).to(device)
#rnn = LSTM(latents, actions, hiddens).to(device)
rnn = RNN(latents, actions, hiddens).to(device)
rnn.load_state_dict(torch.load("./model/MDN_RNN_.pt"))
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
        return self.fc(inputs)

controller = Controller(latents, hiddens, actions).to(device)


def evaluate_control_model(rnn, controller, device):
   
    total_episodes = 10
    time_steps = 50
    s = 0
    cumulative = 0
    cumulative_ = 0
    
    with torch.no_grad():
        while s < total_episodes:
            obs = env.reset()
            
            prev_action = None
            #img = custom_env_render(obs, True)
            action = torch.zeros(1, actions).to(device)
            hidden = [torch.zeros(1, hiddens).to(device) for _ in range(2)]
            action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            action = torch.from_numpy(action).float()
            #reward_ = torch.zeros(1, 1).to(device)
            #hidden = [torch.zeros(1, hiddens).to(device) for _ in range(2)]
            for t in range(time_steps):
                env.render()
                #print(t)
                obs = torch.from_numpy(obs).float()
                rnn_input = torch.cat([obs.to(device), action.to(device)], dim=-1)
                rnn_input= rnn_input.unsqueeze(0).to(device)
                #print(rnn_input)
                
                #out_full,_,_,hidden= rnn.infer(rnn_input,hidden)
                out_full,hidden,_= rnn(rnn_input)
                #print("out_full")
                #print(out_full)
                #print("hidden")
                #print(hidden.shape)
   
                c_in = torch.cat([obs.to("cuda:0").unsqueeze(0) , hidden], dim=-1)

                controller.to(device)
                action = controller(c_in)
                action = action.squeeze(0)
                action = action.detach().to('cpu')
                #action = action.to("cuda:0")
                prob = F.softmax(action.squeeze(0)/0.7, dim=0).numpy()
                #obs, reward, done, info = env.step(action)
                obs, reward, done, info = env.step(prob)
                prev_action = action
                #print(reward)
                reward = torch.Tensor([[reward * (1-int(done))]])
                #reward = torch.where(reward > 0 , 1, 0)
                #action = action.unsqueeze(0).to(device)
                cumulative += reward
                if done:
                    obs = env.reset()
                    break
            
            cumulative_ += cumulative
            s+=1
        cumulative_ = cumulative / s
        return float(cumulative_)




def train_controller(controller,rnn,  mode='real'):
    parameters = controller.parameters()
    es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                  {'popsize': 32})

    start_time = time.time()
    epoch = 0
    best = 0.0
    cur_best = None
    epochs = 50


    while not es.stop():
        print('epoch : {}'.format(epoch))
        solutions = es.ask()
        reward_list = []
        for s_idx, s in enumerate(solutions):
            load_parameters(s, controller)
            if mode == 'real':
                reward = evaluate_control_model(rnn, controller, device)
            elif mode == 'dream':
                reward = evaluate_control_model(rnn, controller, device)

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
            if mode == 'real':
                torch.save(controller.state_dict(), './model/controller.pt')
            elif mode == 'dream':
                torch.save(controller.state_dict(), 'controller_dream.pt')

        epoch += 1
        if epoch > epochs:
            break


    es.result_pretty()

print(train_controller(controller,rnn, 'real'))

