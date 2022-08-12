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
from ENVIRONMENT import socnavenv
from ENVIRONMENT.socnavenv import SocNavEnv
#from draw_socnavenv import SocNavEnv
from tqdm import tqdm
import  UTILITY.utility as utility
from UTILITY.utility import test_data
from UTILITY.utility import get_observation_from_dataset
from UTILITY.utility import transform_processed_observation_into_raw
import time
from tqdm import tqdm

from VAE.vae import VariationalAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 31
input_size = 31

# test_dataset = test_data(batch_size=256)
#with np.load('./data/vae_test_data.npz') as data:
#    test_data = data['observations.npy']
#test_dataset = test_data#[:20000,:]



test_dataset = torch.load('./data/saved_rollout_train.pt')

class VAE_Dataset(torch.utils.data.Dataset):
    def __init__(self, obs_data):
        self.obs_data = obs_data
        
    def __len__(self):
        return len(obs_data)

    def __getitem__(self, idx):
        data = obs_data[idx]
        return data


def flating_obs_data(data):
    imgs = []
    for episode_data in data.values():
        imgs = imgs + episode_data['obs_sequence']
    print('obs_sequence: {}'.format(len(imgs)))
    imgs = torch.stack(imgs, dim=0)
    print('obs_dataset.size :', imgs.size())
    return imgs


obs_data = flating_obs_data(test_dataset)
test_dataset = VAE_Dataset(obs_data)


cv2.namedWindow("input", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("input", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))
cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("output", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))


# load your model architecture/module
#model = Autoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
model = VariationalAutoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
# model = VAE(input_size=input_size, z_dim=z_dim, hidden_dim=200).to(device)
# fill your architecture with the trained weights
model.load_state_dict(torch.load("./MODEL/vae_model.pt"))
model.eval()

#np.random.shuffle(test_dataset)

def code_and_decode(model, data):
    #data = torch.from_numpy(data)
    data = Variable(data, requires_grad=False).to(device)
    with torch.no_grad():
        output,_,_  = model(data)
        output = output.cpu()
    return output 

for i in range(len(test_dataset)):

    input_sample = test_dataset[i]


    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(input_sample)
    image = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "input", dont_draw=True)
    cv2.imshow("input", image)

    # print(image.shape)
    current_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # current_grey1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    input_sample2 = np.reshape(input_sample, (1,input_sample.shape[0]))
    # print('I', input_sample2)
    # print('X', utility.denormalised(utility.normalised(input_sample2)))
    
    #normalised_input = utility.normalised(input_sample2)

    # print('ni', normalised_input)
    output_sample = code_and_decode(model, input_sample2)
    # print('no', normalised_output)
    # print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")
    # print(normalised_output)
    # print(normalised_output.shape)
    #output_sample = utility.denormalised(normalised_output)
    # print('O', output_sample)
    # print('Odddddddddddddddddddddddddddddddddddddddd', output_sample.shape)
    #output_sample = np.array(output_sample)
    output_sample = output_sample.flatten()
    # print(output_sample)
    # print(output_sample.shape)
    # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
    

    robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(output_sample)
    image1 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "output", dont_draw=True)
    cv2.imshow("output", image1)
    next_predicted_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    print(image1.shape)



    merged = cv2.merge([current_grey, current_grey, next_predicted_grey])
    # print(merged.shape, merged.dtype)
    #merged = np.dstack((current_grey, next_grey, next_predicted_grey))
    print(merged.shape, merged.dtype)
    cv2.imshow("Merged", merged)
    # robot_obs_o, goal_obs_o, humans_obs_o = transform_output_into_observation(output)
    # env.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "output")

    for j in tqdm(range(100)):
        k = cv2.waitKey(10)
        if k%255 == 27:
            sys.exit(0)
