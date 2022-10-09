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
        return observation
    
    def discrete_to_continuous_action(self, action:int):
        """
        Function to return a continuous space action for a given discrete action
        """
        if action == 0:
            return np.array([0, 0.25], dtype=np.float32) 
        
        elif action == 1:
            return np.array([0, -0.25], dtype=np.float32) 

        elif action == 2:
            return np.array([1, 0.125], dtype=np.float32) 
        
        elif action == 3:
            return np.array([1, -0.125], dtype=np.float32) 

        elif action == 4:
            return np.array([1, 0], dtype=np.float32)

        elif action == 5:
            return np.array([-1, 0], dtype=np.float32)
        
        elif action == 6:
            return np.array([-0.8, +0.4], dtype=np.float32)

        elif action == 7:
            return np.array([-0.8, -0.4], dtype=np.float32)
        
        else:
            raise NotImplementedError

    def get_action(self, current_state, epsilon):
        self.duelingDQN.load_state_dict(torch.load('./models/duelingdqn_epsilon_decay_rate_0.00015/episode00100000.pth'))
        self.duelingDQN.eval()

            # exploit
        with torch.no_grad():
            q = self.duelingDQN(torch.from_numpy(current_state).reshape(1, -1).float().to(self.device))
            action_discrete = torch.argmax(q).item()
            action_continuous = self.discrete_to_continuous_action(action_discrete)
            return action_continuous, action_discrete



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
            
            prev_action = None
            for t in range(time_steps):
                # env.render()
                obs = agent.preprocess_observation(obs)
                act_continuous, act_discrete = agent.get_action(obs, 0)


                nxt_obs, nxt_reward, done, _ = env.step(act_continuous)
                action = torch.from_numpy(act_continuous).float()
                # obs = torch.from_numpy(obs).float()
                # obs = preprocess_observation(obs)
                obs_sequence.append(obs)
                # nxt_obs_sequence.append(nxt_obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
                obs = nxt_obs
                reward = nxt_reward  

                t+=1
                if done:
                    
        
                    print("Episode [{}/{}] finished after {} timesteps".format(s + 1, self.num_episodes_to_record, t), flush=True)
                    obs = env.reset()
                    break
            self.data_dic[s] = {"obs_sequence":obs_sequence, "action_sequence":action_sequence, 
                        "reward_sequence":reward_sequence, "done_sequence":done_sequence}        
            s+=1
        if self.mode == 'train':
            torch.save(self.data_dic, self.dir_name + 'saved_dqn_rollout_train.pt') 
        elif self.mode  == 'test':
            torch.save(self.data_dic, self.dir_name + 'saved_dqn_rollout_test.pt')
        elif self.mode  == 'val':
            torch.save(self.data_dic, self.dir_name + 'saved_dqn_rollout_validation.pt')



if __name__ == "__main__":
    # env = gym.make("SocNavEnv-v1")


    # env.configure('./configs/env.yaml')





    env = gym.make("SocNavEnv-v1")
    env.configure("./configs/env.yaml")
    env.set_padded_observations(True)

    # config file for the model
    config = "./configs/duelingDQN.yaml"
    input_layer_size = env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]
    agent = DuelingDQNAgent(env, config, input_layer_size=input_layer_size, run_name="duelingDQN_SocNavEnv")
    # agent.eval(num_episodes=10, path=None)
    
    rollout_dic = {}
    rollout_dir = 'Data/'
    train_dataset = Rollout(rollout_dic, rollout_dir,'train', int(total_episodes))
    train_dataset.make_rollout()

    rollout_dic = {}
    rollout_dir = 'Data/'
    val_dataset = Rollout(rollout_dic, rollout_dir,'test', int(total_episodes*0.1))
    val_dataset.make_rollout()


    rollout_dic = {}
    rollout_dir = 'Data/'
    test_dataset = Rollout(rollout_dic, rollout_dir,'val', int(total_episodes*0.1))
    test_dataset.make_rollout()
