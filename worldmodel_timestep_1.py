import numpy as np
import tensorboardX
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim

import scipy.stats as ss
from tensorboardX import SummaryWriter
import gym


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
# device = torch.device( 'cpu')





n_hiddens = 256
n_latents = 47
n_actions = 2
window=16

numberOfActions = 4
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
    # return torch.from_numpy(observation)
    return observation


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

rnn = RNN(n_latents, n_actions, n_hiddens)#.to(device)


ckpt_dir = hp.ckpt_dir#'ckpt'
# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'RobotFrameDatasetsTimestep1window_16', '*me.pth.tar')))[-1] 
ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'RobotFrameDatasetsTimestep1window_16', '015robotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'RobotFrameDatasetsTimestep05window_16', '018robotframe.pth.tar')))[-1] #

# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'mainNonPrePaddedRobotFrameDatasetsTimestep2window_16', '*me.pth.tar')))[-1] 

rnn_state = torch.load(ckpt )#, map_location={'cuda:0': str(device)})
rnn.load_state_dict(rnn_state['model'])
rnn.eval()

class NeuralNetwork(nn.Module):
    '''
    Neural network for continuous action space
    '''
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()

        self.mlp = nn.Linear(input_shape, 32)


        self.mean_l = nn.Linear(32, n_actions)

    def forward(self, x):
        ot_n = self.mlp(x.float())

        # return  F.softmax(self.mean_l(ot_n).squeeze(0).squeeze(0).detach().to('cpu'), dim=0).numpy()   #   torch.tanh(self.mean_l(ot_n))
        # return  F.softmax(self.mean_l(ot_n))   #   torch.tanh(self.mean_l(ot_n))
        return F.softmax(self.mean_l(ot_n).squeeze(0).squeeze(0), dim=0)


def sample_noise(neural_net):
    '''
    Sample noise for each parameter of the neural net
    '''
    nn_noise = []
    for n in neural_net.parameters():
        noise = np.random.normal(size=n.data.numpy().shape)

        nn_noise.append(noise)
    return np.array(nn_noise)


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


def evaluate_neuralnet(nn, env):
    '''
    Evaluate an agent running it in the environment and computing the total reward
    '''
    hiddens = 256

    current_obs = env.reset()
    current_obs = preprocess_observation(current_obs)

    action_ = random.randint(0, 3)
    action = discrete_to_continuous_action(action_)
    # action = np.atleast_2d(action)
    action = torch.from_numpy(action)#.to(self.device)
    # hidden = [torch.zeros(1, 1, hiddens).to(self.device) for _ in range(2)]
    hidden = [torch.zeros(1, 1, hiddens)for _ in range(2)]
    game_reward = 0
    unsqueezed_action = action#.unsqueeze(0)

    while True:
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",torch.from_numpy(current_obs).unsqueeze(0).shape)

        z = torch.from_numpy(current_obs).unsqueeze(0)#.to(self.device)

        unsqueezed_z = z#.unsqueeze(0)
        unsqueezed_action = unsqueezed_action.unsqueeze(0)#.to(self.device)
        #############################################################################################
        with torch.no_grad():
            rnn_input = torch.cat([unsqueezed_z, unsqueezed_action], dim=-1).float()

            current_obs_X_2_,_, _, hidden = rnn.infer(rnn_input.unsqueeze(0),hidden)
            current_obs_X_2_ = current_obs_X_2_.squeeze(0)          
            current_obs_X_2_ = current_obs_X_2_[-1, :]
            # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",current_obs_X_2_.shape)
            # print("111111111111111111111111111111111111111111",unsqueezed_action.shape)
        
            rnn_input_X_2_ = torch.cat([current_obs_X_2_.unsqueeze(0), unsqueezed_action], dim=-1).float()

            _,_, _, hidden = rnn.infer(rnn_input_X_2_.unsqueeze(0),hidden)

        ################################################################################################

        # current_obs = torch.cat((z.unsqueeze(0).unsqueeze(0), hidden[0].unsqueeze(0)),-1)
        current_obs = torch.cat((z.unsqueeze(0), hidden[0]), -1)
        current_obs = current_obs.squeeze(0).squeeze(0)

        # Output of the neural net
        # net_output = nn(preprocess_observation(obs))
        net_output = nn(current_obs)
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",action_)

        # net_output = nn(torch.tensor(preprocess_observation(obs)))   
        # the action is the value clipped returned by the nn
        action_ = np.clip(net_output.data.cpu().numpy().squeeze(), -1, 1)
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",action_)

        max_action = np.argmax(action_)
        # print("sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss",max_action)

        action = discrete_to_continuous_action(max_action)

        new_obs, reward, done, _ = env.step(action)
        unsqueezed_action = torch.from_numpy(action)
        new_obs = preprocess_observation(new_obs)

        current_obs = new_obs

        game_reward += reward

        if done:
            break

    return game_reward

def evaluate_noisy_net(noise, neural_net, env):
    '''
    Evaluate a noisy agent by adding the noise to the plain agent
    '''
    old_dict = neural_net.state_dict()

    # add the noise to each parameter of the NN
    for n, p in zip(noise, neural_net.parameters()):

        p.data += torch.FloatTensor(n * STD_NOISE)

    # evaluate the agent with the noise
    reward = evaluate_neuralnet(neural_net, env)
    # load the previous paramater (the ones without the noise)
    neural_net.load_state_dict(old_dict)

    return reward

def worker(params_queue, output_queue):
    '''
    Function execute by each worker: get the agent' NN, sample noise and evaluate the agent adding the noise. Then return the seed and the rewards to the central unit
    '''

    env = gym.make(ENV_NAME)
    env.configure('./configs/env_timestep_1.yaml')
    env.set_padded_observations(True)

    # actor = NeuralNetwork(env.observation_space.shape[0], env.action_space.shape[0])
    actor = NeuralNetwork(303, 4)

    while True:
        # get the new actor's params
        act_params = params_queue.get()
        if act_params != None:
            # load the actor params
            actor.load_state_dict(act_params)

            # get a random seed
            seed = np.random.randint(1e6)
            # set the new seed
            np.random.seed(seed)

            noise = sample_noise(actor)


            pos_rew = evaluate_noisy_net(noise, actor, env)
            # Mirrored sampling
            neg_rew = evaluate_noisy_net(-noise, actor, env)

            output_queue.put([[pos_rew, neg_rew], seed])
        else:
            break


def normalized_rank(rewards):
    '''
    Rank the rewards and normalize them.
    '''
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm


ENV_NAME = 'SocNavEnv-v1'

# Hyperparameters
STD_NOISE = 0.05
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_ITERATIONS = 100_000

MAX_WORKERS = 4

val_test = True
# VIDEOS_INTERVAL = 100

now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

if __name__ == '__main__':
    # Writer name
    writer_name = 'RNN_1step_ahead_timestep_1{}_{}_{}_{}_{}_{}'.format(ENV_NAME, date_time, str(STD_NOISE), str(BATCH_SIZE), str(LEARNING_RATE), str(MAX_ITERATIONS), str(MAX_WORKERS))
    print('Name:', writer_name)
    best = 0.0
    # Create the test environment
    env = gym.make(ENV_NAME)

    env.configure('./configs/env_timestep_1.yaml')
    env.set_padded_observations(True)

    # Initialize the agent
    actor = NeuralNetwork(303, 4)
    # Initialize the optimizer
    optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir='runs/'+writer_name)

    # Queues to pass and get the variables to and from each processe
    output_queue = mp.Queue(maxsize=BATCH_SIZE)
    params_queue = mp.Queue(maxsize=BATCH_SIZE)

    processes = []

    # Create and start the processes
    for _ in range(MAX_WORKERS):
        p = mp.Process(target=worker, args=(params_queue, output_queue))
        p.start()
        processes.append(p)


    # Execute the main loop MAX_ITERATIONS times
    for n_iter in range(MAX_ITERATIONS):
        it_time = time.time()

        batch_noise = []
        batch_reward = []

        # create the queue with the actor parameters
        for _ in range(BATCH_SIZE):
            params_queue.put(actor.state_dict())

        # receive from each worker the results (the seed and the rewards)
        for i in range(BATCH_SIZE):
            p_rews, p_seed = output_queue.get()

            np.random.seed(p_seed)
            noise = sample_noise(actor)
            batch_noise.append(noise)
            batch_noise.append(-noise)

            batch_reward.append(p_rews[0]) # reward of the positive noise
            batch_reward.append(p_rews[1]) # reward of the negative noise

        # Print some stats
        print(n_iter, 'Mean:',np.round(np.mean(batch_reward), 2), 'Max:', np.round(np.max(batch_reward), 2), 'Time:', np.round(time.time()-it_time, 2))
        writer.add_scalar('reward', np.mean(batch_reward), n_iter)

        # Rank the reward and normalize it
        batch_reward = normalized_rank(batch_reward)


        th_update = []
        optimizer.zero_grad()
        # for each actor's parameter, and for each noise in the batch, update it by the reward * the noise value
        for idx, p in enumerate(actor.parameters()):
            upd_weights = np.zeros(p.data.shape)

            for n,r in zip(batch_noise, batch_reward):
                upd_weights += r*n[idx]

            upd_weights = upd_weights / (BATCH_SIZE*STD_NOISE)
            # put the updated weight on the gradient variable so that afterwards the optimizer will use it
            # print("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",upd_weights)
            # print("3333333333333333333333333333333333333333333333333333333333333333333",p.grad )

            p.grad = torch.FloatTensor( -upd_weights)
            # print("22222222222222222222222222222222222222222222222222222222222",p.grad )

            th_update.append(np.mean(upd_weights))

        # Optimize the actor's NN
        optimizer.step()

        writer.add_scalar('loss', np.mean(th_update), n_iter)

        if n_iter % 50 == 0:        
            torch.save(actor.state_dict(), './models/RNN_1step_ahead_timestep_1.pt')


    # quit the processes
    for _ in range(MAX_WORKERS):
        params_queue.put(None)

    for p in processes:
        p.join()

# tensorboard --logdir content/runs --host localhost