import torch
import torch.nn as nn
import numpy as np
from hparams import VAEHyperParams as hp
from hparams import DQNHyperParams as HP

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
        z = torch.sigmoid(z)
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
        return self.decoder(z),mu , sigma ,z


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


n_hiddens = 256
n_latents = 47
global_step = 0


class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_layers:list, v_net_layers:list, a_net_layers:list) -> None:
        super().__init__()
        # sizes of the first layer in the value and advantage networks should be same as the output of the hidden layer network
        assert(v_net_layers[0]==hidden_layers[-1] and a_net_layers[0]==hidden_layers[-1])
        self.hidden_mlp = MLP(input_size, hidden_layers)
        self.value_network = MLP(v_net_layers[0], v_net_layers[1:])
        self.advantage_network = MLP(a_net_layers[0], a_net_layers[1:])
        

    def forward(self,x):
        x = self.hidden_mlp.forward(x)
        v = self.value_network.forward(x)
        a = self.advantage_network.forward(x)
        q = v + a - torch.mean(a, dim=1, keepdim=True)
        return q

class DuelingDQNAgent:
    def __init__(self, env:gym.Env, config:str, **kwargs) -> None:
        assert(env is not None and config is not None)
        # initializing the env
        self.env = env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # agent variables
        self.input_layer_size = None
        self.hidden_layers = None
        self.v_net_layers = None
        self.a_net_layers = None
        self.buffer_size = None
        self.num_episodes = None
        self.epsilon = None
        self.epsilon_decay_rate = None
        self.batch_size = None
        self.gamma = None
        self.lr = None
        self.polyak_const = None
        self.render = None
        self.min_epsilon = None
        self.save_path = None
        self.render_freq = None
        self.save_freq = None
        self.run_name = None

        # if variables are set using **kwargs, it would be considered and not the config entry
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise NameError(f"Variable named {k} not defined")
        
        # setting values from config file
        self.configure(self.config)

        # declaring the network
        self.duelingDQN = DuelingDQN(self.input_layer_size, self.hidden_layers, self.v_net_layers, self.a_net_layers).to(self.device)
        
        #initializing the fixed targets
        self.fixed_targets = DuelingDQN(self.input_layer_size, self.hidden_layers, self.v_net_layers, self.a_net_layers).to(self.device)
        self.fixed_targets.load_state_dict(self.duelingDQN.state_dict())

        # initalizing the replay buffer
        self.experience_replay = ExperienceReplay(self.buffer_size)

        # variable to keep count of the number of steps that has occured
        self.steps = 0

        if self.run_name is not None:
            self.writer = SummaryWriter('runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.input_layer_size is None:
            self.input_layer_size = config["input_layer_size"]
            assert(self.input_layer_size is not None), f"Argument input_layer_size cannot be None"

        if self.hidden_layers is None:
            self.hidden_layers = config["hidden_layers"]
            assert(self.hidden_layers is not None), f"Argument hidden_layers cannot be None"

        if self.v_net_layers is None:
            self.v_net_layers = config["v_net_layers"]
            assert(self.v_net_layers is not None), f"Argument v_net_layers cannot be None"

        if self.a_net_layers is None:
            self.a_net_layers = config["a_net_layers"]
            assert(self.a_net_layers is not None), f"Argument a_net_layers cannot be None"

        if self.buffer_size is None:
            self.buffer_size = config["buffer_size"]
            assert(self.buffer_size is not None), f"Argument buffer_size cannot be None"

        if self.num_episodes is None:
            self.num_episodes = config["num_episodes"]
            assert(self.num_episodes is not None), f"Argument num_episodes cannot be None"

        if self.epsilon is None:
            self.epsilon = config["epsilon"]
            assert(self.epsilon is not None), f"Argument epsilon cannot be None"

        if self.epsilon_decay_rate is None:
            self.epsilon_decay_rate = config["epsilon_decay_rate"]
            assert(self.epsilon_decay_rate is not None), f"Argument epsilon_decay_rate cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), f"Argument batch_size cannot be None"

        if self.gamma is None:
            self.gamma = config["gamma"]
            assert(self.gamma is not None), f"Argument gamma cannot be None"

        if self.lr is None:
            self.lr = config["lr"]
            assert(self.lr is not None), f"Argument lr cannot be None"

        if self.polyak_const is None:
            self.polyak_const = config["polyak_const"]
            assert(self.polyak_const is not None), f"Argument polyak_const cannot be None"

        if self.render is None:
            self.render = config["render"]
            assert(self.render is not None), f"Argument render cannot be None"

        if self.min_epsilon is None:
            self.min_epsilon = config["min_epsilon"]
            assert(self.min_epsilon is not None), f"Argument min_epsilon cannot be None"

        if self.save_path is None:
            self.save_path = config["save_path"]
            assert(self.save_path is not None), f"Argument save_path cannot be None"

        if self.render_freq is None:
            self.render_freq = config["render_freq"]
            assert(self.render_freq is not None), f"Argument render_freq cannot be None"

        if self.save_freq is None:
            self.save_freq = config["save_freq"]
            assert(self.save_freq is not None), f"Argument save_freq cannot be None"


    def xavier_init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    def preprocess_observation(self, obs):
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
	    
    def discrete_to_continuous_action(self,action:int):
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


    def get_action(self, current_state, epsilon):
        self.duelingDQN.load_state_dict(torch.load('./models/episode00100000.pth'))
        # self.duelingDQN.load_state_dict(torch.load('./models/duelingdqn_epsilon_decay_rate_action_8_v1/episode00100000.pth'))

        self.duelingDQN.eval()

            # exploit
        with torch.no_grad():
            q = self.duelingDQN(torch.from_numpy((current_state.numpy())).reshape(1, -1).float().to(self.device))
            action_discrete = torch.argmax(q).item()
            action_continuous = self.discrete_to_continuous_action(action_discrete)
            return action_continuous, action_discrete

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
# ckpt = sorted(glob.glob(os.path.join(ckpt_dir, 'vae', '*k.pth.tar')))[-1]
# vae_state = torch.load(ckpt)
# model.load_state_dict(vae_state['model'])
model.eval()
# print('Loaded vae ckpt {}'.format(ckpt))
n_actions = 2
n_hiddens = 256
n_latents = 47
rnn = RNN(n_latents, n_actions, n_hiddens).to(device)
# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'rnn', '038k.pth.tar')))[-1]dqnrnn
ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'rnn', '*n.pth.tar')))[-1]
rnn_state = torch.load( ckpt, map_location={'cuda:0': str(device)})
rnn.load_state_dict(rnn_state['model'])
rnn.eval()
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
    
    # print(goal_obs)
    # print("goal_obs")
    # print(humans)
    # print("ddddddddddddddd")

    return goal_obs, humans



def code_and_decode(model, data):
    data = torch.from_numpy(data)

    data = data.view(-1, input_size)
    data = Variable(data, requires_grad=False).to(device)
    with torch.no_grad():
        
        output,_,_ ,z = model(data)
        # print(output)
        
        output = output.squeeze(0)
        # output = output.cpu().numpy()
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
        from collections import deque
        # dq = deque([0,1,2,3,4,5,6,7,8,9], maxlen=10)
        import torch
        # obs = torch.zeros(1, 47).numpy()
        obs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        Observation = deque([obs,obs,obs,obs,obs,obs,obs,obs,obs] ,maxlen=9)
        # action_ = torch.zeros(1, 2).numpy()
        action_ = [0,0]
        Actions = deque([action_,action_,action_,action_,action_,action_,action_,action_,action_] ,maxlen=9)

        while s < self.num_episodes_to_record:
            obs_sequence = []
            nxt_obs_sequence = []
            action_sequence = []
            reward_sequence= []
            done_sequence = []
            obs = env.reset()
            seq_len =9
            prev_reward = 0
            reward = 0
            z = []
            Actions_ = []#torch.zeros(9,47)
            for t in range(time_steps):
                env.render()
                # action_ = np.random.randint(0, 4)

                # action = discrete_to_continuous_action(action_)
                obs = preprocess_observation(obs)
                action, act_discrete = agent.get_action(obs, 0)

                obs_ = obs

                Observation.append(obs.squeeze(0).numpy())
                Actions.append(torch.from_numpy(action).numpy())
                current = Observation[-1]
                
                # goal_obs, humans_obs = transform_processed_observation_into_raw(torch.from_numpy(obs.numpy()))
                goal_obs, humans_obs = transform_processed_observation_into_raw(torch.from_numpy(current))
                Human_list = []
                world_image_new = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                for OBS in humans_obs:
                    Human_list.append(Human(id=1, x=OBS[0], y=OBS[1], theta=OBS[2], width=0.72 ))
                for human in Human_list:
                    human.draw(world_image_new, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
                    
                # to draw the goal of the agent
                cv2.circle(world_image_new, (w2px(goal_obs[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
                #draw the robot
                robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
                robot.draw(world_image_new, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
                input_grey = cv2.cvtColor(world_image_new, cv2.COLOR_BGR2GRAY)

                for i in range(9):
                    z.append(Observation[i])

                    Actions_.append(Actions[i])
                z = np.array(z)
                Actions_= np.array(Actions_)
       
                states = torch.cat([torch.from_numpy(z), torch.from_numpy(Actions_)], dim=-1) # (B, T, vsize+asize)
                states = states.unsqueeze(0).float() .to(device)


                predicted_obs_, _, _ = rnn(states) 
                predicted_obs_ = predicted_obs_.squeeze(0)          
                predicted_obs = predicted_obs_[-1, :]


                #from the observation space get the goal and the human positions with respect to the robots framwork
                goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(predicted_obs.squeeze(0).cpu().detach())
                Human_list = []
                world_image_vae = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                for OBS in humans_obs_o:
                    Human_list.append(Human(id=1, x=OBS[0], y=OBS[1], theta=OBS[2], width=0.72 ))
                for human in Human_list:
                    human.draw(world_image_vae, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
                    
                # to draw the goal of the agent
                cv2.circle(world_image_vae, (w2px(goal_obs_o[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs_o[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
                #draw the robot_
                robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
                robot.draw(world_image_vae, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
                output_grey = cv2.cvtColor(world_image_vae, cv2.COLOR_BGR2GRAY)
                #display image

                nxt_obs, nxt_reward, done, _ = env.step(action)
                action = torch.from_numpy(action).float()

                goal_obs, humans_obs = transform_processed_observation_into_raw(preprocess_observation(nxt_obs).squeeze(0).cpu())
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
                next_timestep_grey = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)

                # cv2.imshow("input", world_image)
                merged = cv2.merge([input_grey, next_timestep_grey, output_grey])
                cv2.imshow("Robot Frame RNN OUTPUT", merged)
                


        
                obs = nxt_obs
                # reward = nxt_reward  
                z = []
                Actions_ = []#torch.zeros(9,47)

                t+=1
                if done:
                    
        
                    print("Episode [{}/{}] finished after {} timesteps".format(s + 1, self.num_episodes_to_record, t), flush=True)
                    obs = env.reset()
                    break
                

if __name__ == "__main__":

    env = gym.make("SocNavEnv-v1")
    env.configure("./configs/env.yaml")
    env.set_padded_observations(True)

    # config file for the model
    config = "./configs/duelingDQN.yaml"
    input_layer_size = env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]
    agent = DuelingDQNAgent(env, config, input_layer_size=input_layer_size, run_name="duelingDQN_SocNavEnv")
    # agent.eval(num_episodes=10, path=None)

    env.configure('./configs/env.yaml')
    env.set_padded_observations(True)

    rollout_dic = {}
    rollout_dir = 'Data/'
    train_dataset = Rollout(rollout_dic, rollout_dir,'train', int(total_episodes))
    train_dataset.make_rollout()



