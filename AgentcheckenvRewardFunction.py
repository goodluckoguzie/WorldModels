import time
import gym
import numpy as np
import stable_baselines3
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from ENVIRONMENT.Socnavenv import SocNavEnv
from ENVIRONMENT import Socnavenv

#import socnavenv1

#from socnavenv1 import SocNavEnv
# from socnavenv import SocNavEnv
from simplejson import load
import os
import pygame

import numpy as np 
import matplotlib.pyplot as plt


pygame.init()
pygame.joystick.init()
controller = pygame.joystick.Joystick(0)


axis_data = { 0:0, 1:0}
button_data = {}
hat_data = {}

env = SocNavEnv()



episodes = 50



def axis_data_to_action(axis_data):
    return np.array([-2*axis_data[1]-1, -axis_data[0]])    

for episode in range(episodes):
    done = False
    obs = env.reset()

    rewards = [0]
    acc_rewards = [0]


    while not done:
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_data[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                hat_data[event.hat] = event.value
        
        # Insert your code on what you would like to happen for each event here!
        action = axis_data_to_action(axis_data)
        print("action", action)
        obs, reward, done, info = env.step(action)
        #print('reward',reward)
        env.render()


        rewards.append(reward)
        acc_rewards.append(acc_rewards[-1]+ reward)

        r = np.array(rewards)
        plt.axhline(y=0., color='k', linestyle='-')
        plt.plot(r) 
        plt.plot(acc_rewards)
        # plt.ylim([-0.02, 0.02])
        # plt.yticks([-0.02, -0.01, 0, 0.01, 0.02])
        plt.pause(Socnavenv.TIMESTEP)


    plt.show()



env.close()