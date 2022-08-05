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
import socnavenv
from socnavenv import SocNavEnv
#from draw_socnavenv import SocNavEnv
from tqdm import tqdm

from utility import test_data
from utility import get_observation_from_dataset
from utility import transform_processed_observation_into_raw
import time
from tqdm import tqdm

import utility
from VAE import Autoencoder, VariationalAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 31
input_size = 31

# test_dataset = test_data(batch_size=256)
with np.load('./data/Train_dataset.npz') as data:
    train_data = data['observations.npy']
test_dataset = train_data[:20000,:]

cv2.namedWindow("input", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("input", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))
cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("output", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))


# load your model architecture/module
#model = Autoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
model = VariationalAutoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
# model = VAE(input_size=input_size, z_dim=z_dim, hidden_dim=200).to(device)
# fill your architecture with the trained weights
model.load_state_dict(torch.load("./saved_model/TVAE_model.pt"))
#model.load_state_dict(torch.load("./saved_model/AE_model.pt"))
model.eval()

np.random.shuffle(test_dataset)


def code_and_decode(model, data):
    data = torch.from_numpy(data)
    data = Variable(data, requires_grad=False).to(device)
    with torch.no_grad():
        output = model(data).cpu()
    return output 

for i in range(len(test_dataset)):

    input_sample = test_dataset[i]


    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(input_sample)
    image = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "input", dont_draw=True)
    cv2.imshow("input", image)

    input_sample2 = np.reshape(input_sample, (1,input_sample.shape[0]))
    print('I', input_sample2)
    print('X', utility.denormalised(utility.normalised(input_sample2)))
    normalised_input = utility.normalised(input_sample2)
    print('ni', normalised_input)
    normalised_output = code_and_decode(model, normalised_input)
    print('no', normalised_output)
    print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")
    print(normalised_output)
    print(normalised_output.shape)
    output_sample = utility.denormalised(normalised_output)
    print('O', output_sample)
    print('Odddddddddddddddddddddddddddddddddddddddd', output_sample.shape)
    #output_sample = np.array(output_sample)
    output_sample = output_sample.flatten()
    print(output_sample)
    print(output_sample.shape)
    print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    

    robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(output_sample)
    image = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "output", dont_draw=True)
    cv2.imshow("output", image)
    
    # robot_obs_o, goal_obs_o, humans_obs_o = transform_output_into_observation(output)
    # env.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "output")

    for j in tqdm(range(100)):
        k = cv2.waitKey(10)
        if k%255 == 27:
            sys.exit(0)