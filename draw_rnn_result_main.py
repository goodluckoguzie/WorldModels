import torch
import torch.nn as nn
import numpy as np
from hparams import RNNHyperParams as hp
from hparams import VAEHyperParams as vhp

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

class RNN(nn.Module):
    def __init__(self, n_latents, n_actions, n_hiddens):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_latents+n_actions, n_hiddens, batch_first=True)
        # target --> next latent (vision)
        self.fc = nn.Linear(n_hiddens, n_latents)

    def forward(self, states):
        h, _ = self.rnn(states)
        y = self.fc(h)
        return y, None, None
    
    def infer(self, states, hidden):
        h, next_hidden = self.rnn(states, hidden) # return (out, hx, cx)
        y = self.fc(h)
        return y, None, None, next_hidden

n_actions = 2
n_hiddens = 256
n_latents = 47
global_step = 0

model = VAE(n_latents,n_hiddens,n_latents).to(device)

ckpt_dir = hp.ckpt_dir

ckpt = sorted(glob.glob(os.path.join(ckpt_dir, 'vae', '*k.pth.tar')))[-1]
vae_state = torch.load(ckpt)
model.load_state_dict(vae_state['model'])
model.eval()
print('Loaded vae ckpt {}'.format(ckpt))       

rnn = RNN(n_latents, n_actions, n_hiddens).to(device)
ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'rnn', '038k.pth.tar')))[-1]
rnn_state = torch.load( ckpt, map_location={'cuda:0': str(device)})
rnn.load_state_dict(rnn_state['model'])
rnn.eval()
print('Loaded rnn_state ckpt {}'.format(ckpt))
seq_len = hp.seq_len
data_path = hp.data_dir 
dataset = GameEpisodeDataset(data_path, seq_len=seq_len)
loader = DataLoader(
    dataset, batch_size=1, shuffle=True, drop_last=True,
    num_workers=hp.n_workers, collate_fn=collate_fn
)

ckpt_dir = os.path.join(hp.ckpt_dir, 'rnn')
sample_dir = os.path.join(ckpt_dir, 'samples')
os.makedirs(sample_dir, exist_ok=True)


test_dataset = loader
device = 'cuda' if torch.cuda.is_available() else 'cpu'


with torch.no_grad():
    for idx, (obs, actions) in enumerate(tqdm(loader, total=len(loader))):
    # for idx, (obs, actions) in enumerate(test_dataset):
        obs, actions = obs.to(device), actions.to(device)

        z_latent,latent_mu, latent_var = model(obs)
        #  # (B*T, vsize)
        z = z_latent.view(-1,seq_len, n_latents) 
        for current_obs in z:
            for actions_obs in actions:


                next_obs_ = current_obs[1:, :]
                z_, actions_ = current_obs[:-1, :], actions_obs[:-1, :]


                states = torch.cat([z_, actions_], dim=-1) # (B, T, vsize+asize)

                states = states.unsqueeze(0)
                predicted_obs_, _, _ = rnn(states) 
                predicted_obs_ = predicted_obs_.squeeze(0)          
                
                next_timestep = next_obs_[-1, :] 
                z_ = z_[-1, :] 
                # nxt_obs = nxt_obs[-1, :]
                predicted_obs = predicted_obs_[-1, :]

                result = np.all((next_timestep.squeeze == 0))
                if result:
                    print('Array contains only 0')
                    continue
                else:
                    print('Array has non-zero items too')


                goal_obs, humans_obs = transform_processed_observation_into_raw(z_.squeeze(0).cpu())
                Human_list = []
                world_image = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                for obs in humans_obs:
                    Human_list.append(Human(id=1, x=obs[0], y=obs[1], theta=obs[2], width=0.72 ))
                for human in Human_list:
                    human.draw(world_image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
                    
                # to draw the goal of the agent
                cv2.circle(world_image, (w2px(goal_obs[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
                #draw the robot
                robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
                robot.draw(world_image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
                current_grey = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)

                cv2.imshow("input", world_image)

                goal_obs, humans_obs = transform_processed_observation_into_raw(next_timestep.squeeze(0).cpu())
                Human_list = []
                world_image = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                for obs in humans_obs:
                    Human_list.append(Human(id=1, x=obs[0], y=obs[1], theta=obs[2], width=0.72 ))
                for human in Human_list:
                    human.draw(world_image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
                    
                # to draw the goal of the agent
                cv2.circle(world_image, (w2px(goal_obs[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
                #draw the robot
                robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
                robot.draw(world_image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
                input_grey = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)

                cv2.imshow("input", world_image)



                # output = code_and_decode(vae, input_sample)
                # output = torch.from_numpy(output)
                goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(predicted_obs.squeeze(0).cpu())
                Human_list = []
                world_image = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                for obs in humans_obs_o:
                    Human_list.append(Human(id=1, x=obs[0], y=obs[1], theta=obs[2], width=0.72 ))
                for human in Human_list:
                    human.draw(world_image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
                    
                # to draw the goal of the agent
                cv2.circle(world_image, (w2px(goal_obs_o[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs_o[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
                #draw the robot_
                robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
                robot.draw(world_image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
                output_grey = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)

                cv2.imshow("output", world_image)

                merged = cv2.merge([current_grey, input_grey, output_grey])
                # # # # print(merged.shape, merged.dtype)
                # # # #merged = np.dstack((current_grey, next_grey, next_predicted_grey))
                # # # print(merged.shape, merged.dtype)
                cv2.imshow("Merged", merged)

                k = cv2.waitKey(1000)
                if k%255 == 27:
                    sys.exit(0) 







if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config file for the model
    config = "./configs/RNN_model.yaml"
        # declaring the network

