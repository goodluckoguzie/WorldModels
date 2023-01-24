import glob
from collections import deque
import os

import random
import sys
from hparams import HyperParams as hp
import gym
from tensorboardX import SummaryWriter
import scipy.stats as ss
from torch import optim
import numpy as np
import tensorboardX
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)


sys.path.append('./gsoc22-socnavenv')
from socnavenv.wrappers import WorldFrameObservations
import socnavenv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device( 'cpu')


n_hiddens = 256
n_latents = 47
n_actions = 2
window = 16

numberOfActions = 4


def preprocess_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    observation = np.array([], dtype=np.float32)
    observation = np.concatenate((observation, obs["goal"].flatten()))
    observation = np.concatenate((observation, obs["humans"].flatten()))
    observation = np.concatenate((observation, obs["laptops"].flatten()))
    observation = np.concatenate((observation, obs["tables"].flatten()))
    observation = np.concatenate((observation, obs["plants"].flatten()))
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
        h, next_hidden = self.rnn(states, hidden)  # return (out, hx, cx)
        y = self.fc(h)
        return y, None, None, next_hidden


rnn = RNN(n_latents, n_actions, n_hiddens).to(device)


ckpt_dir = hp.ckpt_dir  # 'ckpt'
# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'DQN_RobotFrameDatasetsTimestep1window_16', '010DQN_trainedRobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16

ckpt = sorted(glob.glob(os.path.join(
    ckpt_dir, 'RobotFrameDatasetsTimestep1window_16', '015robotframe.pth.tar')))[-1]
# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'RobotFrameDatasetsTimestep05window_16', '018robotframe.pth.tar')))[-1] #

# ckpt  = sorted(glob.glob(os.path.join(ckpt_dir, 'mainNonPrePaddedRobotFrameDatasetsTimestep2window_16', '*me.pth.tar')))[-1]

rnn_state = torch.load(ckpt, map_location={'cuda:0': str(device)})
rnn.load_state_dict(rnn_state['model'])
rnn.eval()


class NeuralNetwork(nn.Module):
    '''
    Neural network for continuous action space
    '''
    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()

        self.mlp = nn.Linear(input_shape, 500)
        self.x11 = nn.Linear(500, 250)
        self.x22 = nn.Linear(250, 32)

        self.mean_l = nn.Linear(32, n_actions)

    def forward(self, x):
        ot_n = self.x22(self.x11 (self.mlp(x.float())))

        # return  F.softmax(self.mean_l(ot_n).squeeze(0).squeeze(0).detach().to('cpu'), dim=0).numpy()   #   torch.tanh(self.mean_l(ot_n))
        # return  F.softmax(self.mean_l(ot_n))   #   torch.tanh(self.mean_l(ot_n))
        return F.softmax(self.mean_l(ot_n).squeeze(0).squeeze(0), dim=0)


def sample_noise(neural_net):
    '''
    Sample noise for each parameter of the neural net
    '''
    nn_noise = []
    for n in neural_net.parameters():
        noise = np.random.normal(size=n.data.detach().cpu().numpy().shape)

        nn_noise.append(noise)
    return np.array(nn_noise)


def discrete_to_continuous_action(action: int):
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
    action00 = discrete_to_continuous_action(action_)
    action0 = torch.from_numpy((discrete_to_continuous_action(0))).unsqueeze(0).to(device)
    action1 = torch.from_numpy((discrete_to_continuous_action(1))).unsqueeze(0).to(device)
    action2 = torch.from_numpy((discrete_to_continuous_action(2))).unsqueeze(0).to(device)
    action3 = torch.from_numpy((discrete_to_continuous_action(3))).unsqueeze(0).to(device)
    # action = np.atleast_2d(action)
    action00 = torch.from_numpy(action00).unsqueeze(0).to(device)
    hidden = [torch.zeros(1, 1, hiddens).to(device) for _ in range(2)]
    hidden_for_action0 = hiddens
    hidden_for_action1 = hiddens
    hidden_for_action2 = hiddens
    hidden_for_action3 = hiddens
    # obs = env.reset()   
    game_reward = 0
    unsqueezed_action = action00.to(device)#.unsqueeze(0)

    while True:
        current_obs = torch.from_numpy(current_obs).unsqueeze(0).to(device)

        unsqueezed_z = current_obs.to(device)#.unsqueeze(0)
        # unsqueezed_action =.to(self.device) unsqueezed_action.unsqueeze(0).to(self.device)
    ###########################################################################################################################                

        with torch.no_grad():
            rnn_input = torch.cat([unsqueezed_z.to(device), unsqueezed_action.to(device)], dim=-1).float()

            _,_, _, hidden = rnn.infer(rnn_input.unsqueeze(0).to(device),hidden)


        #############################################################################################
        with torch.no_grad():
            rnn_input = torch.cat([current_obs, action0], dim=-1).float()

            current_obs_X_2_,_, _, hidden0 = rnn.infer(rnn_input.unsqueeze(0),hidden)
            current_obs_X_2_ = current_obs_X_2_.squeeze(0)          
            current_obs_X_2_ = current_obs_X_2_[-1, :]

        
            rnn_input_X_2_ = torch.cat([current_obs_X_2_.unsqueeze(0), action0], dim=-1).float()

            _,_, _, hidden_for_action0 = rnn.infer(rnn_input_X_2_.unsqueeze(0),hidden0)

        ################################################################################################
        #############################################################################################
        with torch.no_grad():
            rnn_input = torch.cat([current_obs, action1], dim=-1).float()

            current_obs_X_2_,_, _, hidden1 = rnn.infer(rnn_input.unsqueeze(0),hidden)
            current_obs_X_2_ = current_obs_X_2_.squeeze(0)          
            current_obs_X_2_ = current_obs_X_2_[-1, :]

        
            rnn_input_X_2_ = torch.cat([current_obs_X_2_.unsqueeze(0), action1], dim=-1).float()

            _,_, _, hidden_for_action1 = rnn.infer(rnn_input_X_2_.unsqueeze(0),hidden1)

        ################################################################################################
        #############################################################################################
        with torch.no_grad():
            rnn_input = torch.cat([current_obs, action2], dim=-1).float()

            current_obs_X_2_,_, _, hidden2 = rnn.infer(rnn_input.unsqueeze(0),hidden)
            current_obs_X_2_ = current_obs_X_2_.squeeze(0)          
            current_obs_X_2_ = current_obs_X_2_[-1, :]

        
            rnn_input_X_2_ = torch.cat([current_obs_X_2_.unsqueeze(0), action2], dim=-1).float()

            _,_, _, hidden_for_action2 = rnn.infer(rnn_input_X_2_.unsqueeze(0),hidden2)

        ################################################################################################
        #############################################################################################
        with torch.no_grad():
            rnn_input = torch.cat([current_obs, action3], dim=-1).float()

            current_obs_X_2_,_, _, hidden3 = rnn.infer(rnn_input.unsqueeze(0),hidden)
            current_obs_X_2_ = current_obs_X_2_.squeeze(0)          
            current_obs_X_2_ = current_obs_X_2_[-1, :]

        
            rnn_input_X_2_ = torch.cat([current_obs_X_2_.unsqueeze(0), action3], dim=-1).float()

            _,_, _, hidden_for_action3 = rnn.infer(rnn_input_X_2_.unsqueeze(0),hidden3)

        ################################################################################################

        # current_obs = torch.cat((z.unsqueeze(0).unsqueeze(0), hidden[0].unsqueeze(0)),-1)
        # current_obs = torch.cat((z.unsqueeze(0), hidden[0]), -1)
        current_obs = torch.cat((unsqueezed_z.unsqueeze(0), hidden_for_action0[0],hidden_for_action1[0],hidden_for_action2[0],hidden_for_action3[0]), -1)
        current_obs = current_obs.squeeze(0).squeeze(0)
        nn = nn.to(device)

        # sampling an action from the current state
        action_distribution = nn(current_obs).to(device)
        action_ = np.clip(action_distribution.data.cpu().numpy().squeeze(), -1, 1)
        max_action = np.argmax(action_)
        action = discrete_to_continuous_action(max_action)


        # taking a step in the environment
        next_obs, reward, done, info = env.step(action)

        game_reward += reward

        # preprocessing the observation, i.e padding the observation with zeros if it is lesser than the maximum size
        next_obs = preprocess_observation(next_obs)
        next_obs_ = next_obs

        unsqueezed_action = torch.from_numpy(action).unsqueeze(0)#.unsqueeze(0)
        next_obs = torch.from_numpy(next_obs).unsqueeze(0)#.unsqueeze(0)
        # mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state.items()})
        # rnn_input = torch.cat((z, action, reward_), -1).float()
        # out_full, hidden = mdrnn(rnn_input, hidden)


        with torch.no_grad():
            rnn_input = torch.cat([next_obs, unsqueezed_action], dim=-1).float()
            _,_, _, hidden = rnn.infer(rnn_input.unsqueeze(0).to(device),hidden)

        current_obs = next_obs_
        unsqueezed_action = unsqueezed_action


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
        p.data = p.data.cpu()
        p.data += torch.FloatTensor(n * STD_NOISE)
    p.data = p.data.to(device)

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
    actor = NeuralNetwork(1071, 4).to(device)

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
BATCH_SIZE = 100
LEARNING_RATE = 0.001
MAX_ITERATIONS = 100_000

MAX_WORKERS = 8

val_test = True
# VIDEOS_INTERVAL = 100

now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

if __name__ == '__main__':
    # Writer name
    writer_name = 'WORLDMODELRNN_GPU12_{}_{}_{}_{}_{}_{}'.format(ENV_NAME, date_time, str(
        STD_NOISE), str(BATCH_SIZE), str(LEARNING_RATE), str(MAX_ITERATIONS), str(MAX_WORKERS))
    print('Name:', writer_name)
    best = 0.0
    # Create the test environment
    env = gym.make(ENV_NAME)

    env.configure('./configs/env_timestep_1.yaml')
    env.set_padded_observations(True)

    # Initialize the agent
    actor = NeuralNetwork(1071, 4).to(device)
    # Initialize the optimizer
    optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir='runs/'+writer_name)

    queue = mp.Queue()
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

            batch_reward.append(p_rews[0])  # reward of the positive noise
            batch_reward.append(p_rews[1])  # reward of the negative noise

        # Print some stats
        print(n_iter, 'Mean:', np.round(np.mean(batch_reward), 2), 'Max:', np.round(
            np.max(batch_reward), 2), 'Time:', np.round(time.time()-it_time, 2))
        writer.add_scalar('reward', np.mean(batch_reward), n_iter)

        # Rank the reward and normalize it
        batch_reward = normalized_rank(batch_reward)

        th_update = []
        optimizer.zero_grad()
        # for each actor's parameter, and for each noise in the batch, update it by the reward * the noise value

        for idx, p in enumerate(actor.parameters()):
            p = p.cpu()
            upd_weights = np.zeros(p.data.shape)

            for n, r in zip(batch_noise, batch_reward):
                upd_weights += r*n[idx]

            upd_weights = upd_weights / (BATCH_SIZE*STD_NOISE)
            # put the updated weight on the gradient variable so that afterwards the optimizer will use it

            p.grad = torch.FloatTensor(-upd_weights)

            th_update.append(np.mean(upd_weights))

        # Optimize the actor's NN
        optimizer.step()

        writer.add_scalar('loss', np.mean(th_update), n_iter)

        if n_iter % 50 == 0:
            torch.save(actor.state_dict(),'./models/MODELRNN_env_timestGPU.pt')

    # quit the processes
    for _ in range(MAX_WORKERS):
        params_queue.put(None)

    for p in processes:
        p.join()

# tensorboard --logdir content/runs --host localhost
