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
import  os ,glob
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, ExperienceReplay
import sys
sys.path.append('./WorldModels')
import sys
from hparams import RobotFrame_Datasets_Timestep_1 as datas

from hparams import HyperParams as hp



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

        self.input_dim = None
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



        sys.path.append('./WorldModels')
        # from RNN.RNN import LSTM,RNN
  
        # self.data_path = self.data_dir# if not self.extra else self.extra_dir

        self.ckpt_dir = hp.ckpt_dir#'ckpt'
        # self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
        # self.vae_state = torch.load(self.ckpt)
        # self.vae.load_state_dict(self.vae_state['model'])
        # self.vae.eval()
        # print('Loaded vae ckpt {}'.format(self.ckpt))


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_hiddens = 256
        n_latents = 47
        n_actions = 2

        # print(os.getcwd())
        self.vae = VAE(n_latents,n_hiddens,n_latents).to(device)


        self.rnn = RNN(n_latents, n_actions, n_hiddens).to(device)

        # # self.ckpt_dir = hp.ckpt_dir#'ckpt'
        # # self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
        # # self.vae_state = torch.load(self.ckpt)
        # # self.vae.load_state_dict(self.vae_state['model'])
        # # self.vae.eval()
        # # print('Loaded vae ckpt {}'.format(self.ckpt))       
        # # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'DQN_RobotFrameDatasetsTimestep1window_16', '010DQN_trainedRobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
        # # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'mainNonPrePaddedRobotFrameDatasetsTimestep1window_16', '005mainrobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'RobotFrameDatasetsTimestep1window_16', '015robotframe.pth.tar')))[-1] #

        # # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
        # rnn_state = torch.load( self.ckpt, map_location={'cuda:1': str(self.device)})
        # self.rnn.load_state_dict(rnn_state['model'])
        # self.rnn.eval()
        # print('Loaded rnn_state ckpt {}'.format(self.ckpt))



        # declaring the network
        self.duelingDQN = DuelingDQN(self.input_layer_size, self.hidden_layers, self.v_net_layers, self.a_net_layers).to(self.device)

        # initializing using xavier initialization
        self.duelingDQN.apply(self.xavier_init_weights)
        # self.duelingDQN.load_state_dict(torch.load('./models/WORLDMODEL/episode00003650.pth'))

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

        if self.input_dim is None:
            self.input_dim = config["input_dims"]
            assert(self.input_dim is not None), f"Argument input_dims size cannot be None"

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

        # if self.ckpt_dir is None:
        #     self.ckpt_dir = config["ckpt_dir"]
        #     assert(self.ckpt_dir is not None), f"Argument ckpt_dir  cannot be None"

        # if self.ckpt_dir is None:
        #     self.ckpt_dir = config["ckpt_dir"]
        #     assert(self.ckpt_dir is not None), f"Argument ckpt_dir  cannot be None"


    def xavier_init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    # def preprocess_observation(self, obs):
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


    def preprocess_observation(self,obs):
        """
        To convert dict observation to numpy observation
        """
        assert(type(obs) == dict)
        obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
        humans = obs["humans"].flatten()
        for i in range(int(round(humans.shape[0]/(6+7)))):
            index = i*(6+7)
            obs2 = np.concatenate((obs2, humans[index+6:index+6+7]) )
        # return torch.from_numpy(obs2)
        return obs2
    
    def discrete_to_continuous_action(self ,action:int):
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

        # exploit
        with torch.no_grad():
            # q = self.duelingDQN(torch.from_numpy(current_state).reshape(1, -1).float().to(self.device))
            q = self.duelingDQN(current_state.unsqueeze(0)).reshape(1, -1).float().to(self.device)

            action_discrete = torch.argmax(q).item()
            action_continuous = self.discrete_to_continuous_action(action_discrete)
            return action_continuous, action_discrete

    def eval(self, num_episodes=50, path=None):


        sys.path.append('./WorldModels')
        # from RNN.RNN import LSTM,RNN
  
        # self.data_path = self.data_dir# if not self.extra else self.extra_dir

        self.ckpt_dir = hp.ckpt_dir#'ckpt'
        # self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
        # self.vae_state = torch.load(self.ckpt)
        # self.vae.load_state_dict(self.vae_state['model'])
        # self.vae.eval()
        # print('Loaded vae ckpt {}'.format(self.ckpt))

        sys.path.append('./WorldModels')
        # from RNN.RNN import LSTM,RNN
  
        # self.data_path = self.data_dir# if not self.extra else self.extra_dir

        self.ckpt_dir = hp.ckpt_dir#'ckpt'
        # self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
        # self.vae_state = torch.load(self.ckpt)
        # self.vae.load_state_dict(self.vae_state['model'])
        # self.vae.eval()
        # print('Loaded vae ckpt {}'.format(self.ckpt))


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_hiddens = 256
        n_hiddensrnn = 256
        n_latents = 16 #47
        n_actions = 2

        # print(os.getcwd())n_hiddens
        # self.vae = VAE(self.input_dim,n_hiddens,n_latents).to(device)
        self.vae = VAE(self.input_dim,n_hiddens,n_latents).to(device)


        self.rnn = RNN(n_latents, n_actions, n_hiddensrnn).to(device)
        self.ckpt_dir = datas.ckpt_dir#'ckpt'
        self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae_16', '*k.pth.tar')))[-1]

        self.vae_state = torch.load(self.ckpt)
        self.vae.load_state_dict(self.vae_state['model'])
        self.vae.eval()
        print('Loaded vae ckpt {}'.format(self.ckpt))  

 
        # print('Loaded vae ckpt {}'.format(self.ckpt))       
        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'DQN_RobotFrameDatasetsTimestep1window_16', '010DQN_trainedRobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'mainNonPrePaddedRobotFrameDatasetsTimestep1window_16', '005mainrobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
        self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'RobotFrameDatasetsTimestep1window_16_16', '00000056robotframemain16.pth.tar')))[-1] #
        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'RobotFrameDatasetsTimestep1window_199_16', '00000046robotframemain16.pth.tar')))[-1] #
        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'RobotFrameDatasetsTimestep1window_100_16', '00000046robotframemain16.pth.tar')))[-1] #

        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
        rnn_state = torch.load( self.ckpt, map_location={'cuda': str(self.device)})
        self.rnn.load_state_dict(rnn_state['model'])
        self.rnn.eval()
        print('Loaded rnn_state ckpt {}'.format(self.ckpt))

        print('self.input_layer_size',self.input_layer_size)
     

        self.rnn.eval()
        if path is None:
            self.duelingDQN.load_state_dict(torch.load('./models/WM_EXP_A_INPUT_SIZE_VAE_16_RNN_WIND_16_512_128_Exp_1/episode00200000.pth'))

        
        self.duelingDQN.eval()
        total_reward = 0
        hiddens = 256
        #########################################################################################################
        self.reward_per_episode = 0
        self.successive_runs = 0

        self.total_jerk_count = 0
        self.total_velocity = np.array([0.0, 0.0])
        self.total_path_length = 0
        self.total_time = 0
        self.total_idle_time = 0

        # Add counters for the conditions
        self.out_of_map_count = 0
        self.human_collision_count = 0
        self.reached_goal_count = 0
        self.max_steps_count = 0
        self.discomfort_count = 0

        # Initialize lists to store values for each episode
        self.discomfort_counts = []
        self.jerk_counts = []
        self.velocities = []
        self.path_lengths = []
        self.times = []
        self.out_of_maps = []
        self.human_collisions = []
        self.reached_goals = []
        self.max_steps = []
        self.episode_run = []
        self.successive_run = []
        self.episode_reward_ = []
        self.idle_times = []
        ################################################################
        for i in range(num_episodes):
            total_reward = 0

            current_obs = self.env.reset()
            #########################################################
            t = 0
            self.idle_time = 0
            self.path_length = 0
            self.prev_vel = np.array([0.0, 0.0])
            self.jerk = np.array([0.0, 0.0])
            self.prev_acc = np.array([0.0, 0.0])
            self.jerk_count = 0
            self.velocity_sum = np.array([0.0, 0.0])
            self.total_out_of_map = 0
            self.total_discomfort_count = 0
            self.total_human_collision = 0
            self.total_reached_goal = 0
            self.total_max_steps = 0
            self.average_velocity = 0
            self.prev_pos = current_obs["goal"][-2:]
            #################################################################           
            current_obs = self.preprocess_observation(current_obs)
            action_ = random.randint(0, 3)
            action = self.discrete_to_continuous_action(action_)
            # action = np.atleast_2d(action)
            action = torch.from_numpy(action).to(self.device)
            hidden = [torch.zeros(1, 1, hiddens).to(self.device) for _ in range(2)]
            done = False
            self.episode_reward = 0
            self.total_grad_norm = 0
            self.episode_loss = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0
        
            # latent_mu = torch.from_numpy(next_obs)#.unsqueeze(0)
            unsqueezed_action = action#.unsqueeze(0)

            while not done: 
                t = t + 1

                # unsqueezed_action = action.unsqueeze(0)
                z = torch.from_numpy(current_obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    znew,latent_mu, latent_var ,z = self.vae(z) # (B*T, vsize)
                    z = z.to(self.device)

                # current_obs = torch.cat((z.unsqueeze(0).unsqueeze(0), hidden[0].unsqueeze(0)),-1)
                current_obs = torch.cat((z.unsqueeze(0), hidden[0]), -1)
                current_obs = current_obs.squeeze(0).squeeze(0)

                # sampling an action from the current state
                action_continuous, action_discrete = self.get_action(current_obs, self.epsilon)

                # taking a step in the environment
                next_obs, reward, done, info = self.env.step(action_continuous)
                total_reward += reward
                next_obs_raw = next_obs


                self.steps += 1

                # preprocessing the observation,
                next_obs = self.preprocess_observation(next_obs)
                next_obs_ = next_obs

                unsqueezed_action = torch.from_numpy(action_continuous).unsqueeze(0).unsqueeze(0)
                next_obs = torch.from_numpy(next_obs).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    znew,latent_mu, latent_var ,next_obs = self.vae(next_obs.to(self.device)) # (B*T, vsize)
                    next_obs =next_obs.unsqueeze(0)

                    rnn_input = torch.cat([z.unsqueeze(0), unsqueezed_action.to(self.device)], dim=-1).float()
                    # rnn_input = torch.cat([next_obs, unsqueezed_action.to(self.device)], dim=-1).float()
                    _,_, _, hidden = self.rnn.infer(rnn_input.to(self.device),hidden)

                # next_obs = torch.cat((next_obs.to(self.device), hidden[0].to(self.device)), -1)

                

                # storing whether the agent reached the goal
                # if info["REACHED_GOAL"]:
                #     self.has_reached_goal = 1
                #     self.successive_runs += 1                
                # if info["COLLISION"]:
                #     self.has_collided = 1
                #     self.steps = self.env.EPISODE_LENGTH
                ####################################################################################
                if info["REACHED_GOAL"]:
                    self.reached_goal_count += 1
                    self.has_reached_goal = 1
                    self.successive_runs += 1 

                # Update the counters based on the info
                if info["OUT_OF_MAP"]:
                    self.out_of_map_count += 1
                if info["HUMAN_COLLISION"]:
                    self.human_collision_count += 1
                    self.has_collided = 1
                    self.steps = self.env.EPISODE_LENGTH

                # if info["REACHED_GOAL"]:
                #     reached_goal_count += 1
                if info["MAX_STEPS"]:
                    self.max_steps_count += 1
                if info["DISCOMFORT_CROWDNAV"]:
                    self.discomfort_count += 1
                # Calculate idle time
                if action_discrete == 3:
                    self.idle_time += 1

                # Calculate agent velocity, path length and jerk
                current_pos = next_obs_raw["goal"][-2:]
                current_vel = (current_pos - self.prev_pos) / t
                # print("current_pos")
                # print(current_pos)
                # print("")
                # print("current_vel")
                # print(current_vel)
                # print("")
                # print("self.prev_pos")
                # print(self.prev_pos)
                # print("")

                self.path_length += np.linalg.norm(current_pos - self.prev_pos)
                current_acc = (current_vel - self.prev_vel) / t
                self.jerk = (current_acc - (self.prev_vel - self.prev_acc) / t) / t

                # Calculate number of jerks
                if np.linalg.norm(self.jerk) > 0.01:  # Threshold for jerk
                    self.jerk_count += 1

                # Sum the velocities for later calculation of average velocity
                self.velocity_sum += current_vel

                # Update previous values for next calculation
                self.prev_pos = current_pos
                self.prev_vel = current_vel
                self.prev_acc = current_acc
                ######################################################################

                # next_obs = next_obs.squeeze(0).squeeze(0).cpu()
                current_obs = current_obs.squeeze(0).squeeze(0).cpu()
                # storing the current state transition in the replay buffer.
                # self.experience_replay.insert((current_obs, reward, action_discrete, next_obs, done))
                current_obs = next_obs_
                unsqueezed_action = unsqueezed_action.squeeze(0).squeeze(0)

                # self.env.render()
            ###############################################################################
            # Calculate average velocity
            self.velocity_sum = self.velocity_sum / t

            # Compute the magnitude of the velocity
            self.velocity_sum = np.linalg.norm(self.velocity_sum)

            # Update total values
            self.total_jerk_count += (self.jerk_count/t)
            self.total_velocity += self.velocity_sum
            self.total_path_length += self.path_length
            self.total_time += t
            self.total_idle_time += (self.idle_time/t)

            # # Print the metrics
            # print(f"Idle Time: {idle_time * t}s")
            # print(f"Path Length: {path_length}")
            # print(f"Final Velocity: {current_vel}")
            # print(f"Final Jerk: {jerk}")
            self.total_out_of_map += self.out_of_map_count  # /num_episodes
            self.total_discomfort_count += (self.discomfort_count   / t)
            self.total_human_collision += self.human_collision_count  # / num_episodes
            self.total_reached_goal += self.reached_goal_count  # / num_episodes
            self.total_max_steps += self.max_steps_count  # / num_episodes
            # self.reward_per_episode += self.episode_reward  total_reward
            ###############################################################################

            self.episode_reward = total_reward
            print("Episode [{}/{}] finished after {} timesteps".format(i + 1, num_episodes, i), flush=True)
            ########################################################################################################


            # Append the values for each episode
            self.discomfort_counts.append(self.discomfort_count/ t)
            self.jerk_counts.append(self.jerk_count)
            self.velocities.append(self.velocity_sum)
            self.path_lengths.append(self.path_length)
            self.times.append(t)
            self.out_of_maps.append(self.out_of_map_count)
            self.human_collisions.append(self.human_collision_count)
            self.reached_goals.append(self.reached_goal_count)
            self.max_steps.append(self.max_steps_count)
            self.episode_run.append(i)
            self.successive_run.append(self.successive_runs)
            # episode_reward.append(reward_per_episode)
            self.episode_reward_.append(self.episode_reward)
            self.idle_times.append(self.idle_time)
            
            t = 0
            self.successive_runs = 0
            self.out_of_map_count = 0
            self.human_collision_count = 0
            # path_length = 0
            # max_steps_count = 0
            # discomfort_count = 0   
            self.reached_goal_count = 0
            # reward_per_episode = 0
            # idle_time = 0   
            # jerk_count = 0
            # self.velocity_sum = 0
            
        # Calculate averages over all episodes
        avg_jerk_count = self.total_jerk_count / num_episodes
        avg_velocity = self.total_velocity / num_episodes
        avg_path_length = self.total_path_length / num_episodes
        avg_time = self.total_time / num_episodes
        avg_idle_time = self.total_idle_time / num_episodes

        # Calculate averages for the counters
        avg_out_of_map = self.total_out_of_map  /num_episodes
        avg_discomfort_count = self.total_discomfort_count  / num_episodes
        avg_human_collision = self.total_human_collision  / num_episodes
        avg_reached_goal = self.total_reached_goal  / num_episodes
        avg_max_steps = self.total_max_steps  / num_episodes

        # Print the averages
        print(f"Average Idle Time Count: {avg_idle_time}")
        print(f"Average Discomfort Count: {avg_discomfort_count}")
        print(f"Average Jerk Count: {avg_jerk_count}")
        print(f"Average Velocity: {avg_velocity}")
        print(f"Average Path Length: {avg_path_length}")
        print(f"Average Time: {avg_time}s")
        print(f"Average Out of Map: {avg_out_of_map}")
        print(f"Average Human Collision: {avg_human_collision}")
        print(f"Average Reached Goal: {avg_reached_goal}")
        print(f"Average Max Steps : {avg_max_steps}")
        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {self.successive_runs}")
        print(f"Average reward per episode: {np.mean(self.episode_reward_)}")

        import pandas as pd
        import matplotlib.pyplot as plt
        if not os.path.exists("RESULTS"):
            os.makedirs("RESULTS")

     # Calculate averages and store them in a dictionary
        averages_dict = {
            "Average Idle Time Count": avg_idle_time,
            "Average Discomfort Count": avg_discomfort_count,
            "Average Jerk Count": avg_jerk_count,
            "Average Velocity": avg_velocity,
            "Average Path Length": avg_path_length,
            "Average Time": avg_time,
            "Average Out of Map": avg_out_of_map,
            "Average Human Collision": avg_human_collision,
            "Average Reached Goal": avg_reached_goal,
            "Average Max Steps": avg_max_steps,
            "Total episodes run": num_episodes,
            "Total successive runs": self.successive_runs,
            "Average reward per episode": self.reward_per_episode/num_episodes,
        }
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame([averages_dict])

        # Save the DataFrame to a CSV file
        df.to_csv('RESULTS/duelingDQNAveragesresults.csv', index=False)

        # Save the DataFrame to a JSON file
        df.to_json('RESULTS/duelingDQNAveragesresults.json', orient='records')


        # Create a DataFrame with the collected data
        data = pd.DataFrame({
            'Discomfort Counts': self.discomfort_counts,
            'Jerk Counts': self.jerk_counts,
            'Velocities': self.velocities,
            'Path Lengths': self.path_lengths,
            'Times': self.times,
            'Out of Maps': self.out_of_maps,
            'Human Collisions': self.human_collisions,
            'Reached Goals': self.reached_goals,
            'Max Steps': self.max_steps,
            'Episode Run': self.episode_run,
            'Ruccessive Run': self.successive_run,
            'Episode Reward': self.episode_reward_,
            'Idle Times': self.idle_times
        })

        # Save DataFrame to csv
        data.to_csv('RESULTS/duelingDQNresults.csv', index=False)

        # Save DataFrame to json
        data.to_json('RESULTS/duelingDQNresults.json', orient='records')

        # Plot results
        plt.figure(figsize=(10, 6))

        # Plot Idle Times
        plt.subplot(3,4,1)
        plt.plot(self.idle_times)
        plt.title('Idle Times')
        plt.xlabel('Number of Episode')
        plt.ylabel('Idle Time')

        # Plot Successive Run Counts
        plt.subplot(3,4,2)
        plt.plot(self.successive_run)
        plt.title('Successive Run')
        plt.xlabel('Number of Episode')
        plt.ylabel('Successive Run Count')

        # Plot Episode Reward Counts
        plt.subplot(3,4,3)
        plt.plot(self.episode_reward_)
        plt.title('Episode Reward')
        plt.xlabel('Number of Episode')
        plt.ylabel('Episode Reward')

        # Plot Discomfort Counts
        plt.subplot(3,4,4)
        plt.plot(self.discomfort_counts)
        plt.title('Discomfort Counts')
        plt.xlabel('Number of Episode')
        plt.ylabel('Discomfort Count')

        # Plot Jerk Counts
        plt.subplot(3,4,5)
        plt.plot(self.jerk_counts)
        plt.title('Jerk Counts')
        plt.xlabel('Number of Episode')
        plt.ylabel('Jerk Count')

        # Plot Velocities
        plt.subplot(3,4,6)
        plt.plot(self.velocities)
        plt.title('Velocities')
        plt.xlabel('Number of Episode')
        plt.ylabel('Velocity')

        # Plot Path Lengths
        plt.subplot(3,4,7)
        plt.plot(self.path_lengths)
        plt.title('Path Lengths')
        plt.xlabel('Number of Episode')
        plt.ylabel('Path Length')

        # Plot Times
        plt.subplot(3,4,8)
        plt.plot(self.times)
        plt.title('Times')
        plt.xlabel('Number of Episode')
        plt.ylabel('Time')

        # Plot Out of Maps
        plt.subplot(3,4,9)
        plt.plot(self.out_of_maps)
        plt.title('Out of Maps')
        plt.xlabel('Number of Episode')
        plt.ylabel('Out of Map Count')

        # Plot Human Collisions
        plt.subplot(3,4,10)
        plt.plot(self.human_collisions)
        plt.title('Human Collisions')
        plt.xlabel('Number of Episode')
        plt.ylabel('Human Collision Count')

        # Plot Reached Goals
        plt.subplot(3,4,11)
        plt.plot(self.reached_goals)
        plt.title('Reached Goals')
        plt.xlabel('Number of Episode')
        plt.ylabel('Reached Goal Count')

        # Plot Max Steps
        plt.subplot(3,4,12)
        plt.plot(self.max_steps)
        plt.title('Max Steps')
        plt.xlabel('Number of Episode')
        plt.ylabel('Max Step Count')

        plt.tight_layout()
        plt.show()

############################################# HIST ######################################
        fig, axs = plt.subplots(3, 4, figsize=(10, 10))

        # Plot Idle Times
        axs[0, 0].hist(self.idle_times, bins=30)
        axs[0, 0].set_title('Idle Times Histogram')
        axs[0, 0].set_xlabel('Idle Time')
        axs[0, 0].set_ylabel('Frequency')

        # Plot Successive Run Counts
        axs[0, 1].hist(self.successive_run, bins=30)
        axs[0, 1].set_title('Successive Run Histogram')
        axs[0, 1].set_xlabel('Successive Run Count')
        axs[0, 1].set_ylabel('Frequency')

        # Plot Episode Reward Counts
        axs[0, 2].hist(self.episode_reward_, bins=30)
        axs[0, 2].set_title('Episode Reward Histogram')
        axs[0, 2].set_xlabel('Episode Reward')
        axs[0, 2].set_ylabel('Frequency')

        # Plot Discomfort Counts
        axs[0, 3].hist(self.discomfort_counts, bins=30)
        axs[0, 3].set_title('Discomfort Counts Histogram')
        axs[0, 3].set_xlabel('Discomfort Count')
        axs[0, 3].set_ylabel('Frequency')

        # Plot Jerk Counts
        axs[1, 0].hist(self.jerk_counts, bins=30)
        axs[1, 0].set_title('Jerk Counts Histogram')
        axs[1, 0].set_xlabel('Jerk Count')
        axs[1, 0].set_ylabel('Frequency')

        # Plot Velocities
        axs[1, 1].hist(self.velocities, bins=30)
        axs[1, 1].set_title('Velocities Histogram')
        axs[1, 1].set_xlabel('Velocity')
        axs[1, 1].set_ylabel('Frequency')

        # Plot Path Lengths
        axs[1, 2].hist(self.path_lengths, bins=30)
        axs[1, 2].set_title('Path Lengths Histogram')
        axs[1, 2].set_xlabel('Path Length')
        axs[1, 2].set_ylabel('Frequency')

        # Plot Times
        axs[1, 3].hist(self.times, bins=30)
        axs[1, 3].set_title('Times Histogram')
        axs[1, 3].set_xlabel('Time')
        axs[1, 3].set_ylabel('Frequency')

        # Plot Out of Maps
        axs[2, 0].hist(self.out_of_maps, bins=30)
        axs[2, 0].set_title('Out of Maps Histogram')
        axs[2, 0].set_xlabel('Out of Map Count')
        axs[2, 0].set_ylabel('Frequency')

        # Plot Human Collisions
        axs[2, 1].hist(self.human_collisions, bins=30)
        axs[2, 1].set_title('Human Collisions Histogram')
        axs[2, 1].set_xlabel('Human Collision Count')
        axs[2, 1].set_ylabel('Frequency')

        # Plot Reached Goals
        axs[2, 2].hist(self.reached_goals, bins=30)
        axs[2, 2].set_title('Reached Goals Histogram')
        axs[2, 2].set_xlabel('Reached Goal Count')
        axs[2, 2].set_ylabel('Frequency')

        # Plot Max Steps
        axs[2, 3].hist(self.max_steps, bins=30)
        axs[2, 3].set_title('Max Steps Histogram')
        axs[2, 3].set_xlabel('Max Step Count')
        axs[2, 3].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")
    env.configure("./configs/env_timestep_1.yaml")
    env.set_padded_observations(True)
    env.seed(123)



    # rnn = RNN(latents, actions, hiddens).to(self.device)
    # rnn = rnn.float()
    # rnn.load_state_dict(torch.load('./MODEL/model.pt'))
    # rnn.eval()

    # config file for the model
    config = "./configs/duelingDQNRNN.yaml"
    input_layer_size = 274#env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]
    agent = DuelingDQNAgent(env, config, input_layer_size=input_layer_size, run_name="duelingDQN_SocNavEnv")
    agent.eval()
    
