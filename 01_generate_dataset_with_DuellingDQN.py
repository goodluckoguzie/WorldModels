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
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import socnavenv
import gym
import numpy as np
import copy
import random
import torch.optim as optim
import os
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, ExperienceReplay
time_steps = 300
import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp
from hparams import DQN_RobotFrame_Datasets_Timestep_1 as data
import sys
# parser = argparse.ArgumentParser("total_episodes asigning")
# parser.add_argument('--episodes', type=int,
#                     help="Number of episodes.")

# parser.add_argument('--testepisodes', type=int,
#                     help="Number of episodes.") 
# args = parser.parse_args()

# # rollout_dir = 'Data/'
# if not os.path.exists(rollout_dir):
#     os.makedirs(rollout_dir)

# total_episodes = args.episodes

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
        # self.duelingDQN.load_state_dict(torch.load('./models/episode00100000.pth'))
        self.duelingDQN.load_state_dict(torch.load('./models/duelingdqn_TIMESTEP_1_BASELINE_V1/episode00051650.pth'))

        self.duelingDQN.eval()

            # exploit
        with torch.no_grad():
            q = self.duelingDQN(torch.from_numpy((current_state.numpy())).reshape(1, -1).float().to(self.device))
            action_discrete = torch.argmax(q).item()
            action_continuous = self.discrete_to_continuous_action(action_discrete)
            return action_continuous, action_discrete



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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####################################################Padding our Observation#############################################


# def pad_tensor( tensor, pad):
#     pad_size = pad - tensor.size(0)
#     # print("tensor.size(0)",tensor.size())
#     return torch.cat([tensor.to(device), torch.zeros([pad_size, tensor.size(1)]).to(device)], dim=0)
def pad_tensor(t, episode_length, window_length=9, pad_function=torch.zeros):
    pad_size = episode_length - t.size(0) + window_length
    # Add window lenght - 1 infront of the number of obersavtion
    begin_pad       = pad_function([window_length-1, t.size(1)]).to(device)
    # pad the environment with lenght of the episode subtracted from  the total episode length
    episode_end_pad = pad_function([pad_size,      t.size(1)]).to(device)

    return torch.cat([begin_pad,t.to(device),episode_end_pad], dim=0)

###########################################End#############################################################################

def rollout():
    time_steps = data.time_steps

    env = gym.make("SocNavEnv-v1")
    env.configure('./configs/env_timestep_1.yaml')
    # env.configure('./configs/env.yaml')
    env.set_padded_observations(True)

    # seq_len = 300
    max_ep = 5000 #hp.n_rollout
    feat_dir = data.data_dir

    os.makedirs(feat_dir, exist_ok=True)

    for ep in range(max_ep):
        obs_lst, action_lst, reward_lst, next_obs_lst, done_lst = [], [], [], [], []
        obs = env.reset()
        obs = preprocess_observation(obs)   

        done = False
        t = 0

        # while t < time_steps:        
        # while not done and t < (time_steps-2):

        for t in range(time_steps):       
            # env.render()
            action, act_discrete = agent.get_action(obs, 0)
            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess_observation(next_obs)
            action = torch.from_numpy(action)


            np.savez(
                os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep,t)),
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            
            obs_lst.append(obs)
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)
            obs = next_obs
            # print("tooottttttttttttttttttttttttttttttttalllllllllllllllllllllllllllllllll",rew)
            if done:
                print("Episode [{}/{}] finished after {} timesteps".format(ep + 1, max_ep, t), flush=True)
                obs = env.reset()
                obs_lst = torch.stack(obs_lst, dim=0).squeeze(1)
                # print("9999999999999999999999999999999999999999999999999999999999999999999999999", obs_lst.shape )

                # obs_lst = pad_tensor(obs_lst, episode_length=time_steps).cpu().detach().numpy()
                # obs_lst = torch.from_numpy(obs_lst) 
                next_obs_lst = torch.stack(next_obs_lst, dim=0).squeeze(1)
                # next_obs_lst = pad_tensor(next_obs_lst, episode_length=time_steps).cpu().detach().numpy()
                # next_obs_lst = torch.from_numpy(next_obs_lst) 

                done_lst = [int(d) for d in done_lst]
                done_lst = torch.tensor(done_lst).unsqueeze(-1)
                # done_lst = pad_tensor(done_lst, episode_length=time_steps).cpu().detach().numpy()
                # done_lst=torch.from_numpy(done_lst)
                
                action_lst = torch.stack(action_lst, dim=0).squeeze(1)
                # action_lst = pad_tensor(action_lst, episode_length=time_steps).cpu().detach().numpy()
                # action_lst=torch.from_numpy(action_lst)
                
                reward_lst = torch.tensor(reward_lst).unsqueeze(-1)
                # reward_lst = pad_tensor(reward_lst, episode_length=time_steps).cpu().detach().numpy()
                # reward_lst=torch.from_numpy(reward_lst)
                break

        np.savez(
            os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
            obs=np.stack(obs_lst, axis=0), # (T, C, H, W)
            action=np.stack(action_lst, axis=0), # (T, a)
            reward=np.stack(reward_lst, axis=0), # (T, 1)
            next_obs=np.stack(next_obs_lst, axis=0), # (T, C, H, W)
            done=np.stack(done_lst, axis=0), # (T, 1)
        )
        

if __name__ == "__main__":


    env = gym.make("SocNavEnv-v1")
    env.configure("./configs/env_timestep_1.yaml")
    env.set_padded_observations(True)

    # config file for the model
    config = "./configs/duelingDQN.yaml"
    input_layer_size = env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]
    agent = DuelingDQNAgent(env, config, input_layer_size=input_layer_size, run_name="duelingDQN_SocNavEnv")
    # agent.eval(num_episodes=10, path=None)
    
    np.random.seed(hp.seed)
    rollout()
