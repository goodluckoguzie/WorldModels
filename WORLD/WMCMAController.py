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
import sys
from hparams import RobotFrame_Datasets_Timestep_1 as data
sys.path.append('./gsoc22-socnavenv')
import socnavenv
from socnavenv.wrappers import WorldFrameObservations

import cma


ENV_NAME = 'SocNavEnv-v1'
EPISODES_PER_GENERATION = 3
GENERATIONS = 10000
POPULATION_SIZE = 100
SIGMA=0.1
SAVE_PATH = "./models/CMAWM/"

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

class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_shape, 32)
        self.l2 = nn.Linear(32, 32)
        self.lout = nn.Linear(32, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        return self.lout(x)

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

ckpt_dir = hp.ckpt_dir#'ckpt'
# self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
# self.vae_state = torch.load(self.ckpt)
# self.vae.load_state_dict(self.vae_state['model'])
# self.vae.eval()
# print('Loaded vae ckpt {}'.format(self.ckpt))


device = 'cpu'
n_hiddens = 256
n_hiddensrnn = 256
n_latents = 16 #47
n_actions = 2

# print(os.getcwd())n_hiddens
# self.vae = VAE(self.input_dim,n_hiddens,n_latents).to(device)
vae = VAE(23,n_hiddens,n_latents).to(device)


rnn = RNN(n_latents, n_actions, n_hiddensrnn).to(device)
ckpt_dir = data.ckpt_dir#'ckpt'

ckpt = sorted(glob.glob(os.path.join(ckpt_dir, 'vae_16', '*k.pth.tar')))[-1]

vae_state = torch.load(ckpt)
vae.load_state_dict(vae_state['model'])
vae.eval()
print('Loaded vae ckpt {}'.format(ckpt))  


# print('Loaded vae ckpt {}'.format(self.ckpt))       
# self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'DQN_RobotFrameDatasetsTimestep1window_16', '010DQN_trainedRobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
# self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'mainNonPrePaddedRobotFrameDatasetsTimestep1window_16', '005mainrobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'RobotFrameDatasetsTimestep1window_16_16', '00000056robotframemain16.pth.tar')))[-1] #
# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'RobotFrameDatasetsTimestep1window_199_16', '00000046robotframemain16.pth.tar')))[-1] #
# self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'RobotFrameDatasetsTimestep1window_100_16', '00000046robotframemain16.pth.tar')))[-1] #

# self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
rnn_state = torch.load( ckpt, map_location={'cpu': str(device)})
rnn.load_state_dict(rnn_state['model'])
rnn.eval()
print('Loaded rnn_state ckpt {}'.format(ckpt))

# print('self.input_layer_size',input_layer_size)



def preprocess_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
    humans = obs["humans"].flatten()
    for i in range(int(round(humans.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, humans[index+6:index+6+7]) )
    return torch.from_numpy(obs2)
    
# def preprocess_observation_for_prediction(obs):
#     """
#     To convert dict observation to numpy observation
#     """
#     assert(type(obs) == dict)
#     observation = np.array([], dtype=np.float32)
#     observation = np.concatenate((observation, obs["goal"].flatten()) )
#     observation = np.concatenate((observation, obs["humans"].flatten()) )
#     observation = np.concatenate((observation, obs["laptops"].flatten()) )
#     observation = np.concatenate((observation, obs["tables"].flatten()) )
#     observation = np.concatenate((observation, obs["plants"].flatten()) )
#     return observation

def preprocess_observation_for_prediction(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
    humans = obs["humans"].flatten()
    for i in range(int(round(humans.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, humans[index+6:index+6+7]) )
    return torch.from_numpy(obs2)

def discrete_to_continuous_action(action:int):
    if action == 0:
        return np.array([0, 1], dtype=np.float32) 
    # Turning clockwise
    elif action == 1:
        return np.array([0, -1], dtype=np.float32) 
    # # Move forward
    elif action == 2:
        return np.array([1, 0], dtype=np.float32)
    # stop the robot
    elif action == 3:
        return np.array([0, 0], dtype=np.float32)
    else:
        raise NotImplementedError

def evaluate(ann, env, seed, render=False, wait_after_render=False):
    env.seed(seed) # deterministic for demonstration
    obs = env.reset()
    obs = preprocess_observation(obs)
    total_reward = 0
    hiddens = 256
    hidden = [torch.zeros(1, 1, hiddens).to(device) for _ in range(2)]

    while True:
        if render is True:
            env.render()

        # z = torch.from_numpy(obs).unsqueeze(0).to(device)
        z = obs.unsqueeze(0).to(device)
        with torch.no_grad():
            znew,latent_mu, latent_var ,z = vae(z) # (B*T, vsize)
            obs = z.to(device)
            obs = torch.cat((obs.unsqueeze(0), hidden[0]), -1)
            obs = obs.squeeze(0).squeeze(0)

        # Output of the neural net
        net_output = ann(torch.tensor(obs))
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy().argmax()
        action = discrete_to_continuous_action(action)
        next_obs, reward, done, _ = env.step(action)
        next_obs = preprocess_observation(next_obs)
        obs = next_obs.squeeze(0).squeeze(0).cpu()

        with torch.no_grad():
            znew,latent_mu, latent_var ,next_obs = vae(next_obs.unsqueeze(0).to(device)) # (B*T, vsize)
            next_obs =next_obs.unsqueeze(0)

        unsqueezed_action = torch.from_numpy(action).unsqueeze(0).unsqueeze(0)
        # next_obs = torch.from_numpy(next_obs).unsqueeze(0).unsqueeze(0)
        # next_obs = next_obs.unsqueeze(0).unsqueeze(0)
        # rnn_input = torch.cat([z.unsqueeze(0), unsqueezed_action.to(self.device)], dim=-1).float()
        rnn_input = torch.cat([next_obs, unsqueezed_action.to(device)], dim=-1).float()
        _,_, _, hidden = rnn.infer(rnn_input.to(device),hidden)




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
            best = cur_best
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
    # ann = NeuralNetwork(23, 2)
    ann = NeuralNetwork(272, 4)
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
        # while not os.path.exists("start"):
        #     time.sleep(1)

        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        np.random.seed(123)
        now = datetime.datetime.now()
        date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
        writer_name = f'WMcmaC_{ENV_NAME}_pop{POPULATION_SIZE}_k{EPISODES_PER_GENERATION}_sigma{SIGMA}_{date_time}'
        writer = SummaryWriter(log_dir='runs/'+writer_name)

        train_with_cma(GENERATIONS, writer_name)
