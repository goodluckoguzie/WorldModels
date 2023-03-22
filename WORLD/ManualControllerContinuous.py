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

import gym
import cv2
import pickle

from hparams import HyperParams as hp

sys.path.append('./gsoc22-socnavenv')
import socnavenv
from socnavenv.wrappers import WorldFrameObservations

import cma

import pygame
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
################################################
###########   Calibrate joystick   #############
################################################
axes = joystick.get_numaxes()
try:
    with open('joystick_calibration.pickle', 'rb') as f:
        centre, values, min_values, max_values = pickle.load(f)
except:
    centre = {}
    values = {}
    min_values = {}
    max_values = {}
    for axis in range(axes):
        values[axis] = 0.
        centre[axis] = 0.
        min_values[axis] = 0.
        max_values[axis] = 0.
    T = 3.
    print(f'Leave the controller neutral for {T} seconds')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            centre[axis] = joystick.get_axis(axis)
        time.sleep(0.05)
    T = 5.
    print(f'Move the joystick around for {T} seconds trying to reach the max and min values for the axes')
    t = time.time()
    while time.time() - t < T:
        pygame.event.pump()
        for axis in range(axes):
            value = joystick.get_axis(axis)-centre[axis]
            if value > max_values[axis]:
                max_values[axis] = value
            if value < min_values[axis]:
                min_values[axis] = value
        time.sleep(0.05)
    with open('joystick_calibration.pickle', 'wb') as f:
        pickle.dump([centre, values, min_values, max_values], f)



ENV_NAME = 'SocNavEnv-v1'
EPISODES_PER_GENERATION = 1
GENERATIONS = 10000
POPULATION_SIZE = 100
SAVE_PATH = "./models/CMA/"


image = np.zeros((900,900), dtype=np.uint8)

def draw_obs(obs):
    image[:,:] = 0
    goal = obs[6:8]
    humans = obs[8:]
    cx = goal[0]
    cy = goal[1]
    cv2.circle(image, (int(-cy*50+450), int(-cx*50+450)), 10, 180, -1)
    for i in range(int(round(humans.shape[0]/(6+7)))):
        index = int(i*(6+7) + 6)
        cx = humans[index+0]
        cy = humans[index+1]
        cv2.circle(image, (int(-cy*50+450), int(-cx*50+450)), 12, 255, 1)
    cv2.circle(image, (int(450), int(450)), 12, 90, 1)

    cv2.line(image, (0, 450), (900, 450), 255, 3)
    cv2.line(image, (450,0), (450,900), 255, 3)

    cv2.imshow("window_view", image)
    cv2.waitKey(1)


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


def evaluate(ann, env, seed, render=False, wait_after_render=False):
    env.seed(seed) # deterministic for demonstration
    obs = env.reset()
    obs = preprocess_observation(obs)

    total_reward = 0
    while True:
        if render is True:
            env.render()
        pygame.event.pump()
        axis_data = [joystick.get_axis(axis)-centre[axis] for axis in range(4)]
        obs, reward, done, _ = env.step([-axis_data[1], -axis_data[2]])
        obs = preprocess_observation(obs)
        total_reward += reward
        draw_obs(obs)
        if done:
            print(total_reward)
            break
    if wait_after_render:
        env.render()
        time.sleep(5)
    return total_reward


def train_with_cma(generations):
    all_rewards = 0
    for generation in range(generations):
        gen_reward = 0
        seeds = [random.getrandbits(32) for _ in range(EPISODES_PER_GENERATION)]
        for seed_index, seed in enumerate(seeds):
            r = evaluate(None, env, seed, render=True, wait_after_render=True)
            all_rewards += r
            gen_reward += r
            print(f'gen:{gen_reward/(seed_index+1)}   overall:{all_rewards/(seed_index+1+generation*EPISODES_PER_GENERATION)}')




if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    env.configure('./configs/env_timestep_1.yaml')
    env.set_padded_observations(True)

    train_with_cma(GENERATIONS)
