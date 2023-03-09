import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp
import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
import os
import torch


ENV_NAME = 'SocNavEnv-v1'

env = gym.make(ENV_NAME)

env.configure('./configs/env_timestep_1.yaml')
env.set_padded_observations(True)


class NeuralNetwork(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        # self.l1 = nn.Linear(input_shape, 64)
        # self.l2 = nn.Linear(64, 16)
        # self.lout = nn.Linear(16, n_actions)
        self.lout = nn.Linear(input_shape, n_actions)
        
    def forward(self, x):
        # x = F.relu(self.l1(x.float()))
        # x = F.relu(self.l2(x))
        # return self.lout(x)
        return self.lout(x.float())
    
    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p
    
    def set_params(self, x):
        start = 0
        for p in self.parameters():
            e = start + np.prod(p.shape)
            p.data = torch.FloatTensor(x[start:e]).reshape(p.shape)
            start = e


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


def discrete_to_continuous_action(action:int):
    """
    Function to return a continuous space action for a given discrete action
    """
    if action == 0:
        return np.array([0, 1], dtype=np.float32) 
    # Turning clockwise
    elif action == 1:
        return np.array([0, -1], dtype=np.float32) 
    # Turning anti-clockwise and moving forward
    # elif action == 3:
    #     return np.array([1, 0.5], dtype=np.float32) 
    # # Turning clockwise and moving forward
    # elif action == 4:
    #     return np.array([1, -0.5], dtype=np.float32) 
    # # Move forward
    elif action == 2:
        return np.array([1, 0], dtype=np.float32)
    # stop the robot
    elif action == 3:
        return np.array([0, 0], dtype=np.float32)
        # Turning clockwise with a reduced speed and rotation
    # elif action == 7:
    #     return np.array([0.5, 1], dtype=np.float32)
    #     # Turning anti-clockwise with a reduced speed and rotation
    # elif action == 8:
    #     return np.array([0.5, -1], dtype=np.float32)
    
    else:
        raise NotImplementedError

def evaluate(ann, env, view):
    env.seed(0) # deterministic for demonstration
    obs = env.reset()
    obs = preprocess_observation(obs)
    total_reward = 0
    s = 0
    # while True:
    while s < 202:
        # if view is True:

            # if (1 + s) % 5 == 0:

  
            #     env.render()
        # Output of the neural net
        net_output = ann(torch.tensor(obs))

        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy().argmax()

        action = discrete_to_continuous_action(action)

        # print("actionaction",action)
        obs, reward, done, _ = env.step(action)
        obs = preprocess_observation(obs)
        total_reward += reward
        s = s + 1

        if done:
            break
        # print("total_reward",total_reward)

    return total_reward



import cma
np.random.seed(123)


ann = NeuralNetwork(47, 4)


es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1,{'popsize': 100,'seed': 123})

# es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1, {'seed': 123})


def fitness(x, ann, env, visul=False):
    ann.set_params(x)
    return -evaluate(ann, env, view=False)




best = 0
for i in range(100000):
    solutions = np.array(es.ask())
    fits = [fitness(x, ann, env) for x in solutions]

    es.tell(solutions, fits)
    es.disp()
    cur_best = max(fits)
    best_index = np.argmax(fits)
    print("current  value {}...".format(cur_best))

    best_params = solutions[best_index]
    # print('current best reward : {}'.format(cur_best))
    if not best or cur_best >= best:
        best = cur_best
        print("Saving new best with value {}...".format(cur_best))
        d = best_params
        torch.save(ann.state_dict(), 'controller1.pt')
        # if i % 50 == 0:
        #     evaluate(ann, env,view=True)
    def save_model(ann ,path):
        torch.save(ann.state_dict(), path)
    save_path = "./models/WM"
    # saving model
    if (save_path is not None) and ((i+1)%5 == 0):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        try:
            save_model(ann ,os.path.join(save_path, "episode"+ str(i+1).zfill(8) + ".pth"))
        except:
            print("Error in saving model")


print('best reward : {}'.format(best))
