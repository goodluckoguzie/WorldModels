import torch
import torch.nn as nn
import numpy as np
from hparams import VAEHyperParams as hp
# from models import VAE, vae_loss
from torch.utils.data import DataLoader
from data import *
from tqdm import tqdm
import os, sys
from torchvision.utils import save_image
from torch.nn import functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml
# from UTILITY.early_stopping_for_vae import  EarlyStopping
# from models import VAE, vae_loss
import gym
import torch
import argparse
import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
import os
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, ExperienceReplay
import numpy as np
time_steps = 300
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
#from draw_socnavenv import SocNavEnv
from tqdm import tqdm
import gym

RESOLUTION_VIEW = None
window_initialised = False

import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.envs.utils.human import Human
from socnavenv.envs.utils.robot import Robot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 94
input_size = 47



# cv2.namedWindow("input", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("input", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))
# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("output", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))

RESOLUTION_VIEW = 1000
MAP_X = random.randint(10, 10)
MAP_Y = random.randint(10, 10)

RESOLUTION_X = int(1500 * MAP_X/(MAP_X + MAP_Y))
RESOLUTION_Y = int(1500 * MAP_Y/(MAP_X + MAP_Y))
PIXEL_TO_WORLD_X = RESOLUTION_X / MAP_X
PIXEL_TO_WORLD_Y = RESOLUTION_Y / MAP_Y
GOAL_RADIUS = 0.5


def w2px(x, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given x-coordinate in world frame, to get the x-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * (x + (MAP_SIZE / 2)))


def w2py(y, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given y-coordinate in world frame, to get the y-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * ((MAP_SIZE / 2) - y))




def get_observation_from_dataset(dataset, idx):
    sample = dataset[idx,:]
    return transform_processed_observation_into_raw(sample)

def transform_processed_observation_into_raw(sample):

    ind_pos = [6,7]
    goal_obs = [sample[i] for i in ind_pos]
    goal_obs = np.array(goal_obs)
    # print(goal_obs)
    humans = []
    for human_num in range(14, sample.size()[0],13):
        # print(human_num)
        humans.append(sample[human_num:human_num + 3])
    
    # humans_obs = np.array(humans)

    return goal_obs, humans

device = 'cuda' if torch.cuda.is_available() else 'cpu'




class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, input_dims)

        self.input_dims = input_dims
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = self.linear3(z)
        # z = torch.sigmoid(z)
        return z.reshape((-1, self.input_dims))



class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dims,  hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, hidden_dims)
        self.linear4 = nn.Linear(hidden_dims, latent_dims)
        self.linear5 = nn.Linear(hidden_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu =  self.linear4(x)
        sigma = torch.exp(self.linear5(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z ,mu , sigma





class VAE(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_dims, hidden_dims, latent_dims)
        self.decoder = Decoder(input_dims, hidden_dims, latent_dims)

    def forward(self, x):
        z,mu , sigma = self.encoder(x)
        return self.decoder(z),mu , sigma



def vae_loss(recon_x, x, mu, logvar):
    """ VAE loss function """
    recon_loss = nn.MSELoss(size_average=False)
    BCE = recon_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD






n_hiddens = 256
n_latents = 47
global_step = 0
# vae_path = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*.pth.tar')))[-1]
# vae_state = torch.load(vae_path, map_location={'cuda:0': str(device)})

# vae= VAE(n_latents,n_hiddens,n_latents).to(device)

# # # vae = VAE(hp.vsize).to(device)
# vae.load_state_dict(vae_state['model'])
# 
# testset = GameSceneDataset(data_path, training=False)
# test_loader = DataLoader(testset, batch_size=hp.test_batch, shuffle=False, drop_last=True)

model = VAE(n_latents,n_hiddens,n_latents).to(device)

ckpt_dir = hp.ckpt_dir#'ckpt'
ckpt = sorted(glob.glob(os.path.join(ckpt_dir, 'vae', '*k.pth.tar')))[-1]
vae_state = torch.load(ckpt)
model.load_state_dict(vae_state['model'])
model.eval()
print('Loaded vae ckpt {}'.format(ckpt))

# cv2.namedWindow("input", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("input", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))
# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("output", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))

RESOLUTION_VIEW = 1000
MAP_X = random.randint(10, 10)
MAP_Y = random.randint(10, 10)

RESOLUTION_X = int(1500 * MAP_X/(MAP_X + MAP_Y))
RESOLUTION_Y = int(1500 * MAP_Y/(MAP_X + MAP_Y))
PIXEL_TO_WORLD_X = RESOLUTION_X / MAP_X
PIXEL_TO_WORLD_Y = RESOLUTION_Y / MAP_Y
GOAL_RADIUS = 0.5


def w2px(x, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given x-coordinate in world frame, to get the x-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * (x + (MAP_SIZE / 2)))


def w2py(y, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given y-coordinate in world frame, to get the y-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * ((MAP_SIZE / 2) - y))




def get_observation_from_dataset(dataset, idx):
    sample = dataset[idx,:]
    return transform_processed_observation_into_raw(sample)

def transform_processed_observation_into_raw(sample):

    ind_pos = [6,7]
    goal_obs = [sample[i] for i in ind_pos]
    goal_obs = np.array(goal_obs)
    # print(goal_obs)
    humans = []
    for human_num in range(14, sample.size()[0],13):
        # print(human_num)
        humans.append(sample[human_num:human_num + 3])
    
    # humans_obs = np.array(humans)

    return goal_obs, humans






def code_and_decode(model, data):
    data = data.view(-1, input_size)
    # data = torch.from_numpy(data)
    data = Variable(data, requires_grad=False).to(device)
    with torch.no_grad():
        
        output,_,_  = model(data)
        output = output.squeeze(0)
        output = output.cpu().numpy()
        # print(output)
    return output 





parser = argparse.ArgumentParser("total_episodes asigning")
parser.add_argument('--episodes', type=int,
                    help="Number of episodes.")

# parser.add_argument('--testepisodes', type=int,
#                     help="Number of episodes.") 
args = parser.parse_args()

rollout_dir = 'Data/'
if not os.path.exists(rollout_dir):
    os.makedirs(rollout_dir)

total_episodes = args.episodes


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


def preprocess_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    observation = np.array([], dtype=np.float32)
    observation = np.concatenate((observation, obs["goal"].flatten()) )
    # print("sddddddddddddddddd")
    # print(observation.shape)
    # print("hhhhhhhhhhhhhhhhhhhh")
    # print(observation.shape)
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    observation = np.concatenate((observation, obs["humans"].flatten()) )
    observation = np.concatenate((observation, obs["laptops"].flatten()) )
    observation = np.concatenate((observation, obs["tables"].flatten()) )
    observation = np.concatenate((observation, obs["plants"].flatten()) )
    return torch.from_numpy(observation)


class Rollout():
    def __init__(self, data_dic, dir_name,mode, num_episodes_to_record):
        super().__init__()
        self.data_dic = data_dic
        self.dir_name = dir_name
        self.mode = mode
        self.num_episodes_to_record = num_episodes_to_record
        
    def make_rollout(self):
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

        s = 0
        while s < self.num_episodes_to_record:
            obs_sequence = []
            nxt_obs_sequence = []
            action_sequence = []
            reward_sequence= []
            done_sequence = []
            obs = env.reset()

            prev_reward = 0
            reward = 0
            for t in range(time_steps):
                env.render()
                action_ = np.random.randint(0, 4)
                action = discrete_to_continuous_action(action_)
                obs = preprocess_observation(obs)
                obs_ = obs



                goal_obs, humans_obs = transform_processed_observation_into_raw(obs)
                Human_list = []
                world_image_new = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                for obs in humans_obs:
                    Human_list.append(Human(id=1, x=obs[0], y=obs[1], theta=obs[2], width=0.72 ))
                for human in Human_list:
                    human.draw(world_image_new, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
                    
                # to draw the goal of the agent
                cv2.circle(world_image_new, (w2px(goal_obs[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
                #draw the robot
                robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
                robot.draw(world_image_new, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
                input_grey = cv2.cvtColor(world_image_new, cv2.COLOR_BGR2GRAY)


                #send feed the input of the vae with the obsevation (obs)
                output = code_and_decode(model, obs_)
                output = torch.from_numpy(output)

                #from the observation space get the goal and the human positions with respect to the robots framwork
                goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(output)
                Human_list = []
                world_image_vae = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                for obs in humans_obs_o:
                    Human_list.append(Human(id=1, x=obs[0], y=obs[1], theta=obs[2], width=0.72 ))
                for human in Human_list:
                    human.draw(world_image_vae, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
                    
                # to draw the goal of the agent
                cv2.circle(world_image_vae, (w2px(goal_obs_o[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs_o[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
                #draw the robot_
                robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
                robot.draw(world_image_vae, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
                output_grey = cv2.cvtColor(world_image_vae, cv2.COLOR_BGR2GRAY)
                #display image
                merged = cv2.merge([input_grey, input_grey, output_grey])
                cv2.imshow("VAE OUTPUT", merged)
                
                nxt_obs, nxt_reward, done, _ = env.step(action)
                prev_action = action 
                action = torch.from_numpy(action).float()
                # obs = torch.from_numpy(obs).float()
                # obs = preprocess_observation(obs)
                # obs_sequence.append(obs)
                # # nxt_obs_sequence.append(nxt_obs)
                # action_sequence.append(action)
                # reward_sequence.append(reward)
                # done_sequence.append(done)
                obs = nxt_obs
                # reward = nxt_reward  

                t+=1
                if done:
                    
        
                    print("Episode [{}/{}] finished after {} timesteps".format(s + 1, self.num_episodes_to_record, t), flush=True)
                    obs = env.reset()
                    break


if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")


    env.configure('./configs/env.yaml')
    env.set_padded_observations(True)

    rollout_dic = {}
    rollout_dir = 'Data/'
    train_dataset = Rollout(rollout_dic, rollout_dir,'train', int(total_episodes))
    train_dataset.make_rollout()