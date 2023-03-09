import torch
import torch.nn as nn
import torch.nn.functional as F
import socnavenv
import gym
import numpy as np
import os
import torch.optim as optim
import argparse
import yaml
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP

import os ,glob

import sys

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






class ActorCritic(nn.Module):
    def __init__(self, input_dim:int, policy_net_hidden_layers:list, value_net_hidden_layers:list):
        super(ActorCritic, self).__init__()
        self.policy_net = MLP(input_dim, policy_net_hidden_layers)
        self.value_net = MLP(input_dim, value_net_hidden_layers)
        
    def forward(self, state):
        logits = self.policy_net(state)
        value = self.value_net(state)
        return logits, value

class A2CAgent:
    def __init__(self, env:gym.Env, config:str, **kwargs):
        
        # initializing the env
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        # agent variables
        self.input_layer_size = None
        self.policy_net_hidden_layers = None
        self.value_net_hidden_layers = None
        self.num_episodes = None
        self.gamma = None
        self.lr = None
        self.entropy_penalty = None
        self.save_path = None
        self.render = None
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

        # self.ckpt_dir = hp.ckpt_dir#'ckpt'
        # self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]
        # self.vae_state = torch.load(self.ckpt)
        # self.vae.load_state_dict(self.vae_state['model'])
        # self.vae.eval()
        # print('Loaded vae ckpt {}'.format(self.ckpt))       
        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'DQN_RobotFrameDatasetsTimestep1window_16', '010DQN_trainedRobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'mainNonPrePaddedRobotFrameDatasetsTimestep1window_16', '005mainrobotframe.pth.tar')))[-1] #RobotFrameDatasetsTimestep05window_16
        self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'RobotFrameDatasetsTimestep1window_16', '015robotframe.pth.tar')))[-1] #

        # self.ckpt  = sorted(glob.glob(os.path.join(self.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
        rnn_state = torch.load( self.ckpt, map_location={'cuda:0': str(self.device)})
        self.rnn.load_state_dict(rnn_state['model'])
        self.rnn.eval()
        print('Loaded rnn_state ckpt {}'.format(self.ckpt))



        # initializing model
        self.model = ActorCritic(self.input_layer_size, self.policy_net_hidden_layers, self.value_net_hidden_layers).to(self.device)

        # variable to keep count of the number of steps that has occured
        self.steps = 0

        # tensorboard run directory
        if self.run_name is not None:
            self.writer = SummaryWriter('runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
        
        if self.input_layer_size is None:
             self.input_layer_size = config["input_layer_size"]
             assert(self.input_layer_size is not None), "Argument input_layer_size cannot be None"

        if self.policy_net_hidden_layers is None:
             self.policy_net_hidden_layers = config["policy_net_hidden_layers"]
             assert(self.policy_net_hidden_layers is not None), "Argument policy_net_hidden_layers cannot be None"

        if self.value_net_hidden_layers is None:
             self.value_net_hidden_layers = config["value_net_hidden_layers"]
             assert(self.value_net_hidden_layers is not None), "Argument value_net_hidden_layers cannot be None"

        if self.num_episodes is None:
             self.num_episodes = config["num_episodes"]
             assert(self.num_episodes is not None), "Argument num_episodes cannot be None"

        if self.gamma is None:
             self.gamma = config["gamma"]
             assert(self.gamma is not None), "Argument gamma cannot be None"

        if self.lr is None:
             self.lr = config["lr"]
             assert(self.lr is not None), "Argument lr cannot be None"

        if self.entropy_penalty is None:
             self.entropy_penalty = config["entropy_penalty"]
             assert(self.entropy_penalty is not None), "Argument entropy_penalty cannot be None"

        if self.save_path is None:
             self.save_path = config["save_path"]
             assert(self.save_path is not None), "Argument save_path cannot be None"

        if self.render is None:
             self.render = config["render"]
             assert(self.render is not None), "Argument render cannot be None"

        if self.render_freq is None:
             self.render_freq = config["render_freq"]
             assert(self.render_freq is not None), "Argument render_freq cannot be None"

        if self.save_freq is None:
             self.save_freq = config["save_freq"]
             assert(self.save_freq is not None), "Argument save_freq cannot be None"



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
    #     if "walls" in obs.keys():
    #         observation = np.concatenate((observation, obs["walls"].flatten()))
    #     return observation
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

        
    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            logits, _ = self.model.forward(state)
            dist = F.softmax(logits, dim=0)
            probs = Categorical(dist)
            return probs.sample().cpu().detach().item()
    
    def discrete_to_continuous_action(self ,action:int):
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

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def update(self):
        states = torch.FloatTensor(np.array([sars[0] for sars in self.trajectory])).to(self.device)
        actions = torch.LongTensor(np.array([sars[1] for sars in self.trajectory])).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array([sars[2] for sars in self.trajectory])).to(self.device)
        next_states = torch.FloatTensor(np.array([sars[3] for sars in self.trajectory])).to(self.device)
        dones = torch.FloatTensor(np.array([sars[4] for sars in self.trajectory])).view(-1, 1).to(self.device)
            
        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))]).to(self.device) * rewards[j:].to(self.device)) for j in range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        
        logits, values = self.model.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        value_loss = F.mse_loss(values, value_targets.detach())
        
        # compute entropy bonus
        entropy = -torch.mean(torch.sum(dists * torch.log(torch.clamp(dists, 1e-10,1.0)), dim=1))

        # compute policy loss
        advantage = value_targets - values
        policy_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        policy_loss = policy_loss.mean()

        # total loss
        loss = policy_loss + value_loss - self.entropy_penalty*entropy
        self.episode_loss += loss.item()

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        self.total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.losses.append(self.episode_loss)
        self.grad_norms.append(self.total_grad_norm)
        self.successes.append(self.has_reached_goal)
        self.collisions.append(self.has_collided)
        self.steps_to_reach.append(self.steps)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "rewards"), np.array(self.rewards), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "losses"), np.array(self.episode_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.total_grad_norm.cpu()), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(self.has_reached_goal), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(self.has_collided), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"), np.array(self.steps), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("loss / episode", self.episode_loss, episode)
        self.writer.add_scalar("total grad norm / episode", (self.total_grad_norm), episode)
        self.writer.add_scalar("ending in sucess? / episode", self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode", self.has_collided, episode)
        self.writer.add_scalar("Steps to reach goal / episode", self.steps, episode)
        self.writer.flush()


    def train(self):

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.rewards = []
        self.losses = []
        self.grad_norms = []
        self.successes = []
        self.collisions = []
        self.steps_to_reach = []
        
        self.average_reward = 0
        hiddens = 256
        import random

        for i in range(self.num_episodes):
            # resetting the environment before the episode starts
            current_state = self.env.reset()
            
            # preprocessing the observation
            current_obs = self.preprocess_observation(current_state)
            action_ = random.randint(0, 3)
            action = self.discrete_to_continuous_action(action_)
            # action = np.atleast_2d(action)
            action = torch.from_numpy(action).to(self.device)
            hidden = [torch.zeros(1, 1, hiddens).to(self.device) for _ in range(2)]

            # initializing episode related variables
            done = False
            self.episode_reward = 0
            self.total_grad_norm = 0
            self.episode_loss = 0
            self.has_reached_goal = 0
            self.has_collided = 0
            self.steps = 0

            self.trajectory = [] # [[s, a, r, s', done], [], ...]
            unsqueezed_action = action#.unsqueeze(0)
            while not done:
                # unsqueezed_action = action.unsqueeze(0)
                z = torch.from_numpy(current_obs).unsqueeze(0).to(self.device)

                unsqueezed_z = z#.unsqueeze(0)
                unsqueezed_action = unsqueezed_action.unsqueeze(0).to(self.device)
                #############################################################################################
                with torch.no_grad():
                    rnn_input = torch.cat([unsqueezed_z, unsqueezed_action], dim=-1).float()

                    current_obs_X_2_,_, _, hidden = self.rnn.infer(rnn_input.unsqueeze(0),hidden)
                    current_obs_X_2_ = current_obs_X_2_.squeeze(0)          
                    current_obs_X_2_ = current_obs_X_2_[-1, :]
                    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",current_obs_X_2_.shape)
                    # print("111111111111111111111111111111111111111111",unsqueezed_action.shape)
                
                    rnn_input_X_2_ = torch.cat([current_obs_X_2_.unsqueeze(0), unsqueezed_action], dim=-1).float()

                    _,_, _, hidden = self.rnn.infer(rnn_input_X_2_.unsqueeze(0),hidden)

                ################################################################################################

                # current_obs = torch.cat((z.unsqueeze(0).unsqueeze(0), hidden[0].unsqueeze(0)),-1)
                current_obs = torch.cat((z.unsqueeze(0), hidden[0]), -1)
                current_obs = current_obs.squeeze(0).squeeze(0).cpu().detach().numpy()

                # action = self.get_action(current_state)
                action = self.get_action(current_obs)

                action_continuous = self.discrete_to_continuous_action(action)
                

                next_state, reward, done, info = self.env.step(action_continuous)

                # preprocessing the observation,
                next_obs = self.preprocess_observation(next_state)
                next_obs_ = next_obs

                unsqueezed_action = torch.from_numpy(action_continuous).unsqueeze(0).unsqueeze(0)
                next_obs = torch.from_numpy(next_obs).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    rnn_input = torch.cat([next_obs, unsqueezed_action], dim=-1).float()
                    current_obs_X_2_,_, _, hidden = self.rnn.infer(rnn_input.to(self.device),hidden)
                    current_obs_X_2_ = current_obs_X_2_.squeeze(0)          
                    current_obs_X_2_ = current_obs_X_2_[-1, :]
                    rnn_input_X_2_ = torch.cat([current_obs_X_2_.unsqueeze(0).unsqueeze(0), unsqueezed_action.to(self.device)], dim=-1).float()

                    _,_, _, hidden = self.rnn.infer(rnn_input_X_2_,hidden)

                next_obs = torch.cat((next_obs.to(self.device), hidden[0].to(self.device)), -1)


                next_obs = next_obs.squeeze(0).squeeze(0).cpu().detach().numpy()
                                                
                # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",current_state.shape)
                # self.trajectory.append([current_state, action, reward, next_state, done])
                self.trajectory.append([current_obs, action, reward, next_obs, done])
                self.episode_reward += reward
                # current_state = next_state
                current_obs = next_obs_
                unsqueezed_action  = torch.from_numpy(action_continuous).to(self.device)

                self.steps += 1
                if info["REACHED_GOAL"]:
                    self.has_reached_goal = 1
                
                if info["COLLISION"]:
                    self.has_collided = 1
                    self.steps = self.env.EPISODE_LENGTH-1
                
                # rendering if reqd
                if self.render and ((i+1) % self.render_freq == 0):
                    self.env.render()
            
            self.update()

            # plotting using tensorboard
            print(f"Episode {i+1} Reward: {self.episode_reward} Loss: {self.episode_loss}")
            self.plot(i+1)

            # saving model
            if (self.save_path is not None) and ((i+1)%self.save_freq == 0) and self.episode_reward >= self.average_reward:
                if not os.path.isdir(self.save_path):
                    os.makedirs(self.save_path)
                try:
                    self.save_model(os.path.join(self.save_path, "episode"+ str(i+1).zfill(8) + ".pth"))
                except:
                    print("Error in saving model")

            # updating the average reward
            if (i+1) % self.save_freq == 0:
                self.average_reward = 0
            else:
                self.average_reward = ((i%self.save_freq)*self.average_reward + self.episode_reward)/((i%self.save_freq)+1)

    def eval(self, num_episodes, path=None):
        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        
        self.model.eval()

        total_reward = 0
        successive_runs = 0
        for i in range(num_episodes):
            o = self.env.reset()
            o = self.preprocess_observation(o)
            done = False
            while not done:
                act_continuous = self.get_action(o)
                new_state, reward, done, info = self.env.step(act_continuous)
                new_state = self.preprocess_observation(new_state)
                total_reward += reward

                self.env.render()

                if info["REACHED_GOAL"]:
                    successive_runs += 1

                o = new_state

        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {successive_runs}")
        print(f"Average reward per episode: {total_reward/num_episodes}")


if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")
    env.configure("./configs/env_timestep_1.yaml")
    env.set_padded_observations(True)
    # config file for the model
    config = "./configs/a2c1stepaheadpredictivemodel.yaml"
    input_layer_size = env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]+ 256

    # input_layer_size = env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]
    agent = A2CAgent(env, config)
    agent.train()