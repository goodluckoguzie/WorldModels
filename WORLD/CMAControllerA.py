import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os, sys, glob

from hparams import HyperParams as hp
import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
import os
import torch


import cma
import datetime


ENV_NAME = 'SocNavEnv-v1'
EPISODES_PER_GENERATION = 5#0
POPULATION_SIZE = 10#0

np.random.seed(123)
env = gym.make(ENV_NAME)
env.configure('./configs/env_timestep_1.yaml')
env.set_padded_observations(True)


# class NeuralNetwork(nn.Module):

#     def __init__(self, input_shape, n_actions):
#         super(NeuralNetwork, self).__init__()
#         self.l1 = nn.Linear(input_shape, 32)
#         self.l2 = nn.Linear(32, 32)
#         self.lout = nn.Linear(32, n_actions)
        
#     def forward(self, x):
#         x = F.relu(self.l1(x.float()))
#         x = F.relu(self.l2(x))
#         return self.lout(x)
    
#     def get_params(self):
#         p = np.empty((0,))
#         for n in self.parameters():
#             p = np.append(p, n.flatten().cpu().detach().numpy())
#         return p
    
#     def set_params(self, x):
#         start = 0
#         for p in self.parameters():
#             e = start + np.prod(p.shape)
#             p.data = torch.FloatTensor(x[start:e]).reshape(p.shape)
#             start = e

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

# def preprocess_observation(obs):
#     """
#     To convert dict observation to numpy observation
#     """
#     assert(type(obs) == dict)
#     observation = np.array([], dtype=np.float32)
#     observation = np.concatenate((observation, obs["goal"].flatten()) )
#     observation = np.concatenate((observation, obs["humans"].flatten()) )
#     observation = np.concatenate((observation, obs["laptops"].flatten()) )
#     observation = np.concatenate((observation, obs["tables"].flatten()) )
#     observation = np.concatenate((observation, obs["plants"].flatten()) )
#     return torch.from_numpy(observation)

def preprocess_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
    humans = obs["humans"].flatten()
    for i in range(int(round(humans.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, humans[index+6:index+6+7]) )
    return torch.from_numpy(obs2)



def discrete_to_continuous_action(action:int):
    if action == 0:
        return np.array([0, 1], dtype=np.float32) 
    # Turning clockwise
    elif action == 1:
        return np.array([0, -1], dtype=np.float32) 
    # # Move forward
    elif action == 2:
        return np.array([1, 0], dtype=np.float32)
    # stop the robot
    elif action == 3:
        return np.array([0, 0], dtype=np.float32)
    else:
        raise NotImplementedError


def evaluate(ann, env, seed, render=False):
    env.seed(seed) # deterministic for demonstration
    obs = env.reset()
    obs = preprocess_observation(obs)
    total_reward = 0
    while True:
        if render is True:
            env.render()
        # Output of the neural net
        net_output = ann(torch.tensor(obs))
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy().argmax()
        action = discrete_to_continuous_action(action)
        obs, reward, done, _ = env.step(action)
        obs = preprocess_observation(obs)
        total_reward += reward
        if done:
            break
    print("preward",total_reward)

    return total_reward




now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
writer_name = f'cma_{ENV_NAME}_pop{POPULATION_SIZE}_k{EPISODES_PER_GENERATION}_{date_time}'
writer = SummaryWriter(log_dir='runs/'+writer_name)

ann = NeuralNetwork(23, 4)


es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1,{'popsize': POPULATION_SIZE,'seed': 123})

# es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1, {'seed': 123})


def fitness(x, ann, env, seed, view=False):
    ann.set_params(x)
    return -evaluate(ann, env, seed, view)


best = 0
for generation in range(10000):
    seeds = [random.getrandbits(32) for _ in range(EPISODES_PER_GENERATION)]
    # Create population of candidates and evaluate them
    candidates, fitnesses , Maxfitnesses = es.ask(), [],[]
    for candidate in candidates:
        reward = 0
        for seed in seeds:
            # Load new policy parameters to agent.
            # ann.set_params(candidate)
            # Evaluate the agent using stable-baselines predict function
            reward += fitness(candidate, ann, env, seed) 
        average_candidate_reward = reward / EPISODES_PER_GENERATION
        fitnesses.append(average_candidate_reward)
        Maxfitnesses.append(-average_candidate_reward)
    # CMA-ES update
    es.tell(candidates, fitnesses)

    # Display some training infos
    mean_fitness = np.mean(sorted(fitnesses)[:int(0.1 * len(candidates))])
    print("Iteration {:<3} Mean top 10% reward: {:.2f}".format(generation, -mean_fitness))
    cur_best = max(Maxfitnesses)
    best_index = np.argmax(Maxfitnesses)
    print("current  value {}...".format(cur_best))
    writer.add_scalar('mean top 10 reward', -mean_fitness, generation)
    # writer.add_scalar('reward', cur_best, generation)


    best_params = candidates[best_index]
    render_the_test = os.path.exists("render")
    seeds = [random.getrandbits(32) for _ in range(EPISODES_PER_GENERATION)]
    rew = 0
    for seed in seeds:
        rew += evaluate(ann, env, seed, render=render_the_test)
    rew /= EPISODES_PER_GENERATION
    writer.add_scalar('test reward', rew, generation)
    if not best or cur_best >= best:
        best = cur_best
        print("Saving new best with value {}...".format(cur_best))
        d = best_params
        torch.save(ann.state_dict(), 'cat.pt')
    def save_model(ann ,path):
        torch.save(ann.state_dict(), path)
    save_path = "./models/cat"
    # saving model
    if (save_path is not None) and ((generation+1)%5 == 0):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        try:
            save_model(ann, os.path.join(save_path, "_gen_" + str(generation+1).zfill(8) + ".pth"))
        except:
            print("Error in saving model")

print('best reward : {}'.format(best))
