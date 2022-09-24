import sys
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
import torch
from torch import nn
from torch.autograd import Variable
import gym
import numpy
from os import listdir
from RNN.RNN import LSTM,RNN
from VAE.vae import VariationalAutoencoder
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
# from ENVIRONMENT.Socnavenv import SocNavEnv

num_layers = 2
latents = 47
actions = 2
hiddens = 256
epochs = 1
actions = 2
import cma
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
advance_split = 5
rotation_split = 5

def discrete_to_continuous_action(action:int):
    """
    Function to return a continuous space action for a given discrete action
    """
    if action == 0:
        return np.array([0, 0.25], dtype=np.float32) 
    
    elif action == 1:
        return np.array([0, -0.25], dtype=np.float32) 

    elif action == 2:
        return np.array([1, 0.125], dtype=np.float32) 
    
    elif action == 3:
        return np.array([1, -0.125], dtype=np.float32) 

    elif action == 4:
        return np.array([1, 0], dtype=np.float32)

    elif action == 5:
        return np.array([-1, 0], dtype=np.float32)
    
    # elif action == 6:
    #     return np.array([-0.8, +0.4], dtype=np.float32)

    # elif action == 7:
    #     return np.array([-0.8, -0.4], dtype=np.float32)
    
    else:
        raise NotImplementedError

number_of_actions = 6


num_layers = 2
# # rnn = MDN_RNN(latents, actions, hiddens, gaussians).to(device)
# rnn = MDN_RNN(latents, actions, hiddens, gaussians).to(device)
# rnn = LSTM(latents, actions, hiddens,num_layers).to(device)
rnn = RNN(latents, actions, hiddens).to(device)
z_dim = 47
input_size = 47
vae = VariationalAutoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
# vae.load_state_dict(torch.load("./MODEL/vae_model1.pt"))
vae.eval()
vae.float()
num_layers = 2
latents = 47
actions = 2
hiddens = 256
epochs = 1
actions = 2


def preprocess_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    observation = np.array([], dtype=np.float32)
    observation = np.concatenate((observation, obs["goal"].flatten()) )
    observation = np.concatenate((observation, obs["humans"].flatten()) )
    observation = np.concatenate((observation, obs["laptops"].flatten()) )
    observation = np.concatenate((observation, obs["tables"].flatten()) )
    observation = np.concatenate((observation, obs["plants"].flatten()) )
    return torch.from_numpy(observation)




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
    total_episodes = 100
    time_steps = 300
    s = 0
    cumulative = 0
    cumulative_ = 0

    rnn = RNN(latents, actions, hiddens).to(device)
    rnn = rnn.float()
    rnn.load_state_dict(torch.load('./MODEL/model.pt'))
    rnn.eval()


    while s < total_episodes:
        obs = env.reset()
        # #action = torch.zeros(1, actions).to(device)
        # action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        action = random.randint(0, 5)
        action = discrete_to_continuous_action(action)
        action = np.atleast_2d(action)
        action = torch.from_numpy(action).to(device)
        prev_action = None
        for t in range(time_steps): 
            #env.render()
            obs = preprocess_observation(obs)
            # obs = torch.from_numpy(obs)
            z = obs.unsqueeze(0).to(device)

            # _, mu, log_var = vae(obs.unsqueeze(0).to(device))
            # sigma = log_var.exp()
            # eps = torch.randn_like(sigma)
            # z = eps.mul(sigma).add_(mu)

            unsqueezed_action = action.unsqueeze(0)
            unsqueezed_z = z.unsqueeze(0)
            # mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state.items()})
            # rnn_input = torch.cat((z, action, reward_), -1).float()
            # out_full, hidden = mdrnn(rnn_input, hidden)
            with torch.no_grad():
                rnn_input = torch.cat([unsqueezed_z, unsqueezed_action], dim=-1).float()
                print("dddddddddddddddddddddddddddddddddddddd",rnn_input.shape)
                _,_, hidden = rnn(rnn_input)
      
            c_in = torch.cat((z.unsqueeze(0).unsqueeze(0), hidden[0].unsqueeze(0)),-1)

            action_distribution = controller(c_in)            
            max_action = np.argmax(action_distribution)            
            action = discrete_to_continuous_action(max_action)

            obs, reward, done, _ = env.step(action)

            action = torch.from_numpy(action)
            reward = torch.Tensor([[reward * (1-int(done))]])
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
    epochs = 8
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

if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")
    env.configure('./configs/env.yaml')
    train_controller(controller, vae, rnn)
    