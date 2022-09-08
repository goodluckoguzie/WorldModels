from telnetlib import FORWARD_X
import time
import gym
import numpy as np
import stable_baselines3
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC
from ENVIRONMENT.Socnavenv import SocNavEnv
# from ENVIRONMENT import Socnavenv

#import socnavenv1

#from socnavenv1 import SocNavEnv
# from socnavenv import SocNavEnv
from simplejson import load
import os
import pickle
import pygame

import numpy as np 
import matplotlib.pyplot as plt


pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)


axis_data = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
ROTATION_AXIS = 1
FORWARD_AXIS = 2

env = SocNavEnv()



episodes = 50

################################################
###########   Calibrate joystick   #############
################################################
axes = joystick.get_numaxes()
try:
    with open('joystick_calibration.json', 'rb') as f:
        centre, values, min_values, max_values = pickle.load(f)
except:
    centre = {}
    values = {}
    min_values = {}
    max_values = {}
    for axis in range(joystick.get_numaxes()):
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
    with open('joystick_calibration.json', 'wb') as f:
        pickle.dump([centre, values, min_values, max_values], f)
print(min_values)
print(max_values)

def axis_data_to_action(axis_data):
    return np.array([-2*axis_data[ROTATION_AXIS]-1, -axis_data[FORWARD_AXIS]])

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

for episode in range(episodes):
    done = False
    obs = env.reset()


    fig.clear()
    rewards = [0,0]
    acc_rewards = [0,0]
    p1, = plt.plot(rewards) 
    p2, = plt.plot(acc_rewards)    
    fig.show()
    for _ in range(50):
        env.render()
        p1, = plt.plot(rewards)
        p2, = plt.plot(acc_rewards)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)
    while not done:
        pygame.event.pump()
        for axis in range(axes):
            axis_data[axis] = joystick.get_axis(axis)-centre[axis]
        
        # Insert your code on what you would like to happen for each event here!
        action = axis_data_to_action(axis_data)
        print("action", action)
        obs, reward, done, info = env.step(action)
        #print('reward',reward)
        env.render()


        rewards.append(reward)
        acc_rewards.append(acc_rewards[-1] + reward)
        # steps.append(steps[-1] + 1)

        # p1.set_xdata(steps)
        p1, = plt.plot(rewards) 
        p2, = plt.plot(acc_rewards)    
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # plt.axhline(y=0., color='k', linestyle='-')
        # plt.ylim([-2, 2])
    for _ in range(25):
        p1, = plt.plot(rewards)
        p2, = plt.plot(acc_rewards)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)

    plt.pause(1)




env.close()