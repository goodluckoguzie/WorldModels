import os
import sys
import time
import glob
import random
import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import gym

from hparams import HyperParams as hp

sys.path.append('./gsoc22-socnavenv')
import socnavenv

import cma


ENV_NAME = 'SocNavEnv-v1'
EPISODES_PER_GENERATION = 10
GENERATIONS = 10000
POPULATION_SIZE = 100
SIGMA=0.1
SAVE_PATH = "./models/CMA/"


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_shape, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.lout = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.elu(self.l1(x.float()))
        x = F.elu(self.l2(x))
        x = F.elu(self.l3(x))
        return torch.tanh(self.lout(x))

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
    #observation = np.array([], dtype=np.float32)
    #observation = np.concatenate((observation, obs["goal"].flatten()) )
    #observation = np.concatenate((observation, obs["humans"].flatten()) )
    #observation = np.concatenate((observation, obs["laptops"].flatten()) )
    #observation = np.concatenate((observation, obs["tables"].flatten()) )
    #observation = np.concatenate((observation, obs["plants"].flatten()) )
    obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
    humans = obs["humans"].flatten()
    for i in range(int(round(humans.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, humans[index+6:index+6+7]) )
    return torch.from_numpy(obs2)


def evaluate(ann, env, seed, render=False, wait_after_render=False):
    env.seed(seed) # deterministic for demonstration
    obs = env.reset()
    obs = preprocess_observation(obs)
    prev_obs = obs
    total_reward = 0
    while True:
        if render is True:
            env.render()
        # Output of the neural net
        cat = torch.cat((prev_obs, obs), 0)
        net_output = ann(cat)
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy()

        prev_obs = torch.tensor(obs)
        obs, reward, done, _ = env.step(action)
        obs = preprocess_observation(obs)
        total_reward += reward
        if done:
            break
    if wait_after_render:
        for i in range(2):
            env.render()
            time.sleep(1)
    return total_reward


def fitness(candidate, env, seed, render=False):
    ann.set_params(candidate)
    return -evaluate(ann, env, seed, render)


def train_with_cma(generations, writer_name):
    es = cma.CMAEvolutionStrategy(len(ann.get_params())*[0], SIGMA, {'popsize': POPULATION_SIZE, 'seed': 123})
    best = 0
    for generation in range(generations):
        seeds = [random.getrandbits(32) for _ in range(EPISODES_PER_GENERATION)]
        # Create population of candidates and evaluate them
        candidates, fitnesses , Maxfitnesses = es.ask(), [],[]
        for candidate in candidates:
            reward = 0
            for seed in seeds:
                # Evaluate the agent using stable-baselines predict function
                reward += fitness(candidate, env, seed, render=False) 
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
        # print("current  value {}...".format(cur_best))
        writer.add_scalar('mean top 10 reward', -mean_fitness, generation)
        # writer.add_scalar('reward', cur_best, generation)

        best_params = candidates[best_index]
        render_the_test = os.path.exists("render")
        seeds = [random.getrandbits(32) for _ in range(EPISODES_PER_GENERATION)]
        test_rew = 0
        # GOODLUCK: You forgot this
        ann.set_params(best_params)
        for seed in seeds:
            test_rew += evaluate(ann, env, seed, render=render_the_test)
        test_rew /= EPISODES_PER_GENERATION
        writer.add_scalar('test reward', test_rew, generation)
        # Save model if it's best
        if not best or test_rew >= best:
            best = test_rew
            print("Saving new best with value {}...".format(best))
            torch.save(ann.state_dict(), writer_name+'_BEST.pth')
        # Saving model every 
        if (generation+1)%5 == 0:
            try:
                torch.save(ann.state_dict(), os.path.join(SAVE_PATH, writer_name+"_gen_" + str(generation+1).zfill(8) + ".pth"))
            except:
                print("Error in saving model")

    print('best reward : {}'.format(best))


if __name__ == '__main__':
    ann = NeuralNetwork(23*2, 2)
    env = gym.make(ENV_NAME)
    env.configure('./configs/env_timestep_1.yaml')
    env.set_padded_observations(True)

    if len(sys.argv)>2 and sys.argv[1] == '-test':
        ann.load_state_dict(torch.load(sys.argv[2]))
        accumulated_reward = 0
        done = 0
        while True:
            reward = evaluate(ann, env, seed=random.getrandbits(32), render=True, wait_after_render=True)
            accumulated_reward += reward
            done += 1
            print(f'Reward: {reward}    (avg:{accumulated_reward/done})')
    else:
        while not os.path.exists("start"):
            time.sleep(1)

        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        np.random.seed(123)
        now = datetime.datetime.now()
        date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
        writer_name = f'cmaC_{ENV_NAME}_pop{POPULATION_SIZE}_k{EPISODES_PER_GENERATION}_sigma{SIGMA}_{date_time}'
        writer = SummaryWriter(log_dir='runs/'+writer_name)

        train_with_cma(GENERATIONS, writer_name)
