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
from RNN.RNN import Rnn, RNN,LSTM
from UTILITY import utility 
from UTILITY.utility import test_data
from UTILITY.utility import get_observation_from_dataset
from UTILITY.utility import transform_processed_observation_into_raw
import time
from tqdm import tqdm
import cma


parser = argparse.ArgumentParser("total epochs asigning")
parser.add_argument('--epochs', type=int,
                    help="Number of epochs.")

args = parser.parse_args()      

env = SocNavEnv()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

action_rnn = 2
latents = 31
hiddens = 256
reward = 1
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
total_episodes = args.epochs


rnn = RNN(latents, action_rnn, hiddens).to(device)
rnn.load_state_dict(torch.load("./MODEL/MDN_RNN_window.pt"))
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


def evaluate_control_model(rnn, controller, device):
   
  
    time_steps = 50
    s = 0
    cumulative = 0
    cumulative_ = 0
    
    with torch.no_grad():
        while s < total_episodes:
            obs = env.reset()
            
            
            prev_action = None
            action = torch.zeros(1, action_rnn).to(device)
            hidden = [torch.zeros(1, hiddens).to(device) for _ in range(2)]
            action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            action = torch.from_numpy(action).float()

            for t in range(time_steps):
                env.render()
                obs = torch.from_numpy(obs).float()
                rnn_input = torch.cat([obs.to(device), action.to(device)], dim=-1)
                rnn_input= rnn_input.unsqueeze(0).to(device)
                out_full,hidden,_= rnn(rnn_input)

                c_in = torch.cat([obs.to("cuda:0").unsqueeze(0) , hidden], dim=-1)
                # controller.to(device)  # Check if this line is necessary
                action_distribution = controller(c_in)
                print(action_distribution)
                
                max_action = np.argmax(action_distribution)
                action = action_list[max_action]
                obs = obs.cpu().numpy()
                obs, reward, done, info = env.step(action)
                action = torch.from_numpy(action).to(device).float()
                #prev_action = action
                #print(reward)
                #reward = torch.Tensor([[reward * (1-int(done))]])
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
                                  {'popsize': 28})

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
                torch.save(controller.state_dict(), './MODEL/controller1.pt')
            elif mode == 'dream':
                torch.save(controller.state_dict(), 'controller_dream.pt')

        epoch += 1
        if epoch > epochs:
            break


    es.result_pretty()

print(train_controller(controller,rnn, 'real'))
