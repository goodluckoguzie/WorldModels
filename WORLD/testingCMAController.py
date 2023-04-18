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
import cma


ENV_NAME = 'SocNavEnv-v1'

env = gym.make(ENV_NAME)

env.configure('./configs/env_timestep_1.yaml')
env.set_padded_observations(True)


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_shape, 32)
        self.l2 = nn.Linear(32, 32)
        self.lout = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        return self.lout(x)


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

def evaluate(ann, env):
    seed=random.getrandbits(32)
    env.seed(seed)
    obs = env.reset()
    obs = preprocess_observation(obs)
    total_reward = 0
    # ann.load_state_dict(torch.load('./models/WM_modelhalfnetwork/episode00005680.pth'))
    # ann.load_state_dict(torch.load('./models/CMA1/cma_SocNavEnv-v1_pop100_k3_15_16.49.18_gen_00001365.pth'))
    # ann.load_state_dict(torch.load('./models/CMA2/cma_SocNavEnv-v1_pop100_k3_15_16.50.7_gen_00001385.pth'))
    ann.load_state_dict(torch.load('./models/CMA3/cma_SocNavEnv-v1_pop100_k3_15_16.50.24_gen_00001425.pth'))
    # ann.load_state_dict(torch.load('./WM_modelC.pt'))
    s = 0
    R = 0

    while True:

        env.render()

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
        # print('Total reward',R)
        # AverageReward = AverageReward + R

    return total_reward



np.random.seed(123)
ann = NeuralNetwork(47, 4)
AverageReward = 0
for episode in range(500):
    episode = episode + 1
    Rew = evaluate(ann,env)
    AverageReward = Rew + AverageReward

    # print("Total reward = ",Rew)
    # print("episodes = ",episode)
print("Average reward after 50 episodes = ",AverageReward/episode)