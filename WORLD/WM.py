

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp
import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, ExperienceReplay
import sys
from hparams import RobotFrame_Datasets_Timestep_1 as data
import time
import datetime

from tensorboardX import SummaryWriter
now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)


from hparams import HyperParams as hp

ENV_NAME = 'SocNavEnv-v1'

env = gym.make(ENV_NAME)

env.configure('./configs/env_timestep_1.yaml')
env.set_padded_observations(True)
writer_name = 'WM_MLP_F100_{}_{}'.format(ENV_NAME, date_time)

writer = SummaryWriter(log_dir='runs/'+writer_name)

from hparams import HyperParams as hp
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'



##################################################################################################
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
        self.N.loc = self.N.loc.to(device)#.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)#.cuda()
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
        return self.decoder(z),mu , sigma,z

    def dcode(self, x):
        d = self.decoder(x)
        return d
        # return self.decoder(z),mu , sigma ,z


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





sys.path.append('./WorldModels')
# from RNN.RNN import LSTM,RNN

# self.data_path = self.data_dir# if not self.extra else self.extra_dir

ckpt_dir = hp.ckpt_dir#'ckpt'
# self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
# self.vae_state = torch.load(self.ckpt)
# self.vae.load_state_dict(self.vae_state['model'])
# self.vae.eval()
# print('Loaded vae ckpt {}'.format(self.ckpt))


n_hiddens = 256
n_hiddensrnn = 64
n_latents = 20 #47
n_actions = 2
n_of_action = 4
input_dim = 47
input_layer_size = 84
# print(os.getcwd())n_hiddens
vae = VAE(input_dim,n_hiddens,n_latents).to(device)


rnn = RNN(n_latents, n_actions, n_hiddensrnn).to(device)
ckpt_dir = data.ckpt_dir#'ckpt'

ckpt = sorted(glob.glob(os.path.join(ckpt_dir, 'vae', '*k.pth.tar')))[-1]

vae_state = torch.load(ckpt)
vae.load_state_dict(vae_state['model'])
vae.eval()
print('Loaded vae ckpt {}'.format(ckpt))  


# print('Loaded vae ckpt {}'.format(self.ckpt))       
# self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'DQN_RobotFrameDatasetsTimestep1window_16', '010DQN_trainedRobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
# self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'mainNonPrePaddedRobotFrameDatasetsTimestep1window_16', '005mainrobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'RobotFrameDatasetsTimestep1window_16', '010robotframe.pth.tar')))[-1] #

# self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
# rnn_state = torch.load( ckpt, map_location={'cuda': str(device)})
rnn_state = torch.load( ckpt, map_location={'cuda': str(device)})
rnn.load_state_dict(rnn_state['model'])
rnn.eval()
print('Loaded rnn_state ckpt {}'.format(ckpt))

print('self.input_layer_size',input_dim)

####################################################################################################


# class NeuralNetwork(nn.Module):

#     def __init__(self, input_shape, n_actions):
#         super(NeuralNetwork, self).__init__()
#         # self.l1 = nn.Linear(input_shape, 64)
#         # self.l2 = nn.Linear(64, 16)
#         # self.lout = nn.Linear(16, n_actions)
#         self.lout = nn.Linear(input_shape, n_actions)
        
#     def forward(self, x):
#         # x = F.relu(self.l1(x.float()))
#         # x = F.relu(self.l2(x))
#         # return self.lout(x)
#         return self.lout(x.float())
        
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
    '''
    Neural network for continuous action space
    '''
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()

        self.mlp = nn.Linear(input_shape, 128)


        self.mean_l = nn.Linear(128, n_actions)

    def forward(self, x):
        ot_n = self.mlp(x.float())

        # return  F.softmax(self.mean_l(ot_n).squeeze(0).squeeze(0).detach().to('cpu'), dim=0).numpy()   #   torch.tanh(self.mean_l(ot_n))
        # return  F.softmax(self.mean_l(ot_n))   #   torch.tanh(self.mean_l(ot_n))
        return F.softmax(self.mean_l(ot_n).squeeze(0).squeeze(0), dim=0)    

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
    observation = np.array([], dtype=np.float32)
    observation = np.concatenate((observation, obs["goal"].flatten()) )
    observation = np.concatenate((observation, obs["humans"].flatten()) )
    observation = np.concatenate((observation, obs["laptops"].flatten()) )
    observation = np.concatenate((observation, obs["tables"].flatten()) )
    observation = np.concatenate((observation, obs["plants"].flatten()) )
    return torch.from_numpy(observation)


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

def evaluate(ann, env, view):




    '''
    Evaluate an agent running it in the environment and computing the total reward
    '''
    total_reward = 0
    hiddens = 64

    current_obs = env.reset()
    current_obs = preprocess_observation(current_obs)

    action_ = random.randint(0, 3)
    action = discrete_to_continuous_action(action_)
    # action = np.atleast_2d(action)
    action = torch.from_numpy(action).to(device)
    hidden = [torch.zeros(1, 1, hiddens).to(device) for _ in range(2)]
    unsqueezed_action = action#.unsqueeze(0)
    # current_obs = torch.from_numpy(current_obs)

    s = 0
    # while True:
    while s < 202:
        s  = s +1

        # unsqueezed_action = action.unsqueeze(0)
        # z = torch.from_numpy(current_obs).unsqueeze(0).to(device)
        z = current_obs.unsqueeze(0).to(device)

        unsqueezed_z = z#.unsqueeze(0)
        unsqueezed_action = unsqueezed_action.unsqueeze(0).to(device)

        with torch.no_grad():
            znew,latent_mu, latent_var ,z = vae(unsqueezed_z)# (B*T, vsize)
            z = z.to(device)
            # rnn_input = torch.cat([unsqueezed_z, unsqueezed_action], dim=-1).float()


            rnn_input = torch.cat([z, unsqueezed_action], dim=-1).float()
            _,_, _, hidden = rnn.infer(rnn_input.unsqueeze(0),hidden)
            current_obs = torch.cat((z.unsqueeze(0), hidden[0]), -1)
            current_obs = current_obs.squeeze(0).squeeze(0).to(device)

        # Output of the neural net
        net_output = ann(current_obs).to(device)

        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy().argmax()

        action = discrete_to_continuous_action(action)
        # print("sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss",max_action)

        next_obs, reward, done, _ = env.step(action)
        next_obs = preprocess_observation(next_obs)
        
        unsqueezed_action = torch.from_numpy(action)#.unsqueeze(0)#.unsqueeze(0)
        current_obs = next_obs.squeeze(0)

        total_reward += reward


        if done:
                break
                e = 0

    return total_reward



import cma
np.random.seed(123)


ann = NeuralNetwork(84, 4).to(device)

# print(len(ann.get_params()) )
es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1,{'popsize': 100,'seed': 123})

# es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1, {'seed': 123})


def fitness(x, ann, env, visul=False):
    ann.set_params(x)
    ann = ann.to(device)
    return -evaluate(ann, env, view=False)


best = 0
for iteration in range(10000):
    # Create population of candidates and evaluate them
    candidates, fitnesses , Maxfitnesses = es.ask(), [],[]
    for candidate in candidates:
        # Load new policy parameters to agent.
        # ann.set_params(candidate)
        # Evaluate the agent using stable-baselines predict function
        reward = fitness(candidate, ann, env) 
        fitnesses.append(reward)
        Maxfitnesses.append(-reward)
    # CMA-ES update
    es.tell(candidates, fitnesses)
    # Display some training infos
    mean_fitness = np.mean(sorted(fitnesses)[:int(0.1 * len(candidates))])
    print("Iteration {:<3} Mean top 10% reward: {:.2f}".format(iteration, -mean_fitness))
    
    cur_best = max(Maxfitnesses)
    best_index = np.argmax(Maxfitnesses)
    print("current  value {}...".format(cur_best))

    writer.add_scalar('Mean top 10 reward', -mean_fitness, iteration)
    writer.add_scalar('reward', cur_best, iteration)


    best_params = candidates[best_index]
    rew = evaluate(ann, env, view=True)
    writer.add_scalar('test reward', rew, iteration)
    # print('current best reward : {}'.format(cur_best))
    if not best or cur_best >= best:
        best = cur_best
        print("Saving new best with value {}...".format(cur_best))
        d = best_params
        torch.save(ann.state_dict(), 'WM_MLP_F100.pt')
        # if i % 50 == 0:
        #     evaluate(ann, env,view=True)

    def save_model(ann ,path):
        torch.save(ann.state_dict(), path)
    save_path = "./models/WM_MLP_F100"
    # saving model
    if (save_path is not None) and ((iteration+1)%5 == 0):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        try:
            save_model(ann ,os.path.join(save_path, "episode"+ str(iteration+1).zfill(8) + ".pth"))
        except:
            print("Error in saving model")

print('best reward : {}'.format(best))
