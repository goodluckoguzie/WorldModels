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
from hparams import RobotFrame_Datasets_Timestep_1 as data

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
        self.ckpt_dir = data.ckpt_dir#'ckpt'

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
            # self.duelingDQN.load_state_dict(torch.load('./models/Uncertainty_EXP_A_INPUT_SIZE_VAE_16_RNN_WIND_16_512_128_Exp_1/episode00192500.pth'))
            self.duelingDQN.load_state_dict(torch.load('./models/ADTSPHSM_EXP_A_INPUT_SIZE_VAE_16_RNN_WIND_16_512_128_Exp_1/episode00199500.pth'))

        
        self.duelingDQN.eval()

        total_reward = 0
        successive_runs = 0

        hiddens = 256
        for i in range(num_episodes):
            current_obs = self.env.reset()
            current_obs = self.preprocess_observation(current_obs)
            action_ = random.randint(0, 3)
            action00 = self.discrete_to_continuous_action(action_)
            action0 = torch.from_numpy((self.discrete_to_continuous_action(0))).unsqueeze(0).to(self.device)
            action1 = torch.from_numpy((self.discrete_to_continuous_action(1))).unsqueeze(0).to(self.device)
            action2 = torch.from_numpy((self.discrete_to_continuous_action(2))).unsqueeze(0).to(self.device)
            action3 = torch.from_numpy((self.discrete_to_continuous_action(3))).unsqueeze(0).to(self.device)
            # action = np.atleast_2d(action)
            action00 = torch.from_numpy(action00).unsqueeze(0).to(self.device)
            hidden = [torch.zeros(1, 1, hiddens).to(self.device) for _ in range(2)]
            hidden_for_action0 = hidden
            hidden_for_action1 = hidden
            hidden_for_action2 = hidden
            hidden_for_action3 = hidden

            done = False
            self.episode_reward = 0
            self.total_grad_norm = 0
            self.episode_loss = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0
            unsqueezed_action = action00.to(self.device)

            while not done: 

            
                current_obs = torch.from_numpy(current_obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    znew,latent_mu, latent_var ,current_obs = self.vae(current_obs) # (B*T, vsize)
                unsqueezed_z = current_obs.to(self.device)
                old_obs = unsqueezed_z

                current_obs = torch.cat((unsqueezed_z.unsqueeze(0), hidden_for_action0[0],hidden_for_action1[0],hidden_for_action2[0],hidden_for_action3[0]), -1)
                current_obs = current_obs.squeeze(0).squeeze(0)

                # sampling an action from the current state
                action_continuous, action_discrete = self.get_action(current_obs, self.epsilon)

                # taking a step in the environment
                next_obs, reward, done, info = self.env.step(action_continuous)

                # incrementing total steps
                self.steps += 1

                # preprocessing the observation,
                next_obs = self.preprocess_observation(next_obs)
                next_obs_ = next_obs


                unsqueezed_action = torch.from_numpy(action_continuous).unsqueeze(0).to(self.device)
                next_obs = torch.from_numpy(next_obs).unsqueeze(0).to(self.device)


                with torch.no_grad():

                    znew,latent_mu, latent_var ,next_obs = self.vae(next_obs) 
                    next_obs =next_obs
                unsqueezed_z     = next_obs

                #########################################################################################################################################
                with torch.no_grad():

                    rnn_input = torch.cat([next_obs, unsqueezed_action], dim=-1).float()
                    # rnn_input = torch.cat([old_obs, unsqueezed_action.to(self.device)], dim=-1).float()

                    _,_, _, hidden = self.rnn.infer(rnn_input.unsqueeze(0).to(self.device),hidden)
                
                #############################################################################################

                if (unsqueezed_action==action0).all() : #unsqueezed_action.squeeze(0) != action0.squeeze(0):
                # if (unsqueezed_action!=action0).all() : #unsqueezed_action.squeeze(0) != action0.squeeze(0):

                    with torch.no_grad():


                        rnn_input = torch.cat([next_obs, action0], dim=-1).float()

                        _,_, _, hidden_for_action0 = self.rnn.infer(rnn_input.unsqueeze(0),hidden)

                else:
                    hidden_for_action0 = hidden

                    ################################################################################################
                if (unsqueezed_action==action1).all():
                # if (unsqueezed_action!=action1).all():

                    with torch.no_grad():
                        rnn_input = torch.cat([next_obs, action1], dim=-1).float()

                        _,_, _, hidden_for_action1 = self.rnn.infer(rnn_input.unsqueeze(0),hidden)
 
                else:
                                hidden_for_action1 = hidden
                    #############################################################################################
                if (unsqueezed_action==action2).all():
                # if (unsqueezed_action!=action2).all():

                    with torch.no_grad():
                        rnn_input = torch.cat([next_obs, action2], dim=-1).float()

                        _,_, _, hidden_for_action2 = self.rnn.infer(rnn_input.unsqueeze(0),hidden)

                else:
                                hidden_for_action2 = hidden
                    #############################################################################################
                if (unsqueezed_action==action3).all():
                # if (unsqueezed_action!=action3).all():

                    with torch.no_grad():
                        rnn_input = torch.cat([next_obs, action3], dim=-1).float()

                        _,_, _, hidden_for_action3 = self.rnn.infer(rnn_input.unsqueeze(0),hidden)

                else:
                                hidden_for_action3 = hidden
                    ################################################################################################

                next_obs = torch.cat((unsqueezed_z.unsqueeze(0), hidden_for_action0[0],hidden_for_action1[0],hidden_for_action2[0],hidden_for_action3[0]), -1)




                # rendering 
                if self.render and ((i+1) % self.render_freq == 0):
                    self.env.render()

                # storing the rewards
                self.episode_reward += reward

                # storing whether the agent reached the goal
                if info["REACHED_GOAL"]:
                    self.has_reached_goal = 1
                    successive_runs += 1

                
                if info["HUMAN_COLLISION"]:
                    self.has_collided = 1
                    self.steps = self.env.EPISODE_LENGTH


                next_obs = next_obs.squeeze(0).squeeze(0).cpu()
                current_obs = current_obs.squeeze(0).squeeze(0).cpu()
                # storing the current state transition in the replay buffer.

                current_obs = next_obs_
                # unsqueezed_action = unsqueezed_action.squeeze(0).squeeze(0)
                unsqueezed_action = unsqueezed_action

                self.env.render()
            total_reward += self.episode_reward
            print("Episode [{}/{}] finished after {} timesteps".format(i + 1, num_episodes, i), flush=True)

        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {successive_runs}")
        print(f"Average reward per episode: {total_reward/num_episodes}")

if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")
    env.seed(123) # deterministic for demonstration

    env.configure("./configs/env_timestep_1.yaml")
    env.set_padded_observations(True)



    # rnn = RNN(latents, actions, hiddens).to(self.device)
    # rnn = rnn.float()
    # rnn.load_state_dict(torch.load('./MODEL/model.pt'))
    # rnn.eval()

    # config file for the model
    config = "./configs/ActionDependentTwoStepPredictiveHiddenStateModel.yaml"
    input_layer_size = 1040 #env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]+hiddens
    # input_layer_size = 94#env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]

    agent = DuelingDQNAgent(env, config, input_layer_size=input_layer_size, run_name="WORLDMODEL")
    agent.train()
    
    
