import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
from UTILITY import utility

import numpy as np
from collections import deque, namedtuple

# For visualization
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from RNN.RNN import LSTM,RNN

from VAE.vae import VariationalAutoencoder
z_dim = 31
input_size = 31
vae = VariationalAutoencoder(input_dims=input_size, hidden_dims=200, latent_dims=z_dim).to(device)
vae.load_state_dict(torch.load("./MODEL/vae_model.pt"))
vae.eval()
vae.float()
latents = 31
actions = 2
hiddens = 256
gaussians = 5
epochs = 10
actions = 2
num_layers = 2


advance_split = 5
rotation_split = 5

advance_grid, rotation_grid = np.meshgrid(
    np.linspace(-1, 1, advance_split),
    np.linspace(-1, 1, rotation_split))

action_list = np.hstack((
    advance_grid.reshape(( advance_split*rotation_split, 1)),
    rotation_grid.reshape((advance_split*rotation_split, 1))))
number_of_actions = action_list.shape[0]


# rnn = MDN_RNN(latents, actions, hiddens, gaussians).to(device)
# rnn = LSTM(latents, actions, hiddens,num_layers).to(device)
# # rnn = RNN(latents, actions, hiddens).to(device)
rnn = RNN(latents, actions, hiddens).to(device)
# rnn = LSTM(latents, actions, hiddens,num_layers).to(device)
# rnn = RNN(latents, actions, hiddens).to(device)
rnn.load_state_dict(torch.load("./MODEL/model.pt"))
rnn = rnn.float()
# rnn.load_state_dict(torch.load("./MODEL/MDN_RNN_window.pt"))
rnn.eval()

from ENVIRONMENT.Socnavenv import SocNavEnv 
env = SocNavEnv()

# from ENVIRONMENT.sOcnavenv import DiscreteSocNavEnv 
# env = DiscreteSocNavEnv()
# env = gym.make('LunarLander-v2')SocNavEnv()
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network




class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
 
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:


            max_action = np.argmax(action_values.cpu().data.numpy())
            # action = action_list[max_action]
            # return np.argmax(action_values.cpu().data.numpy())
            return max_action#action
        else:
            max_action = random.choice(np.arange(self.action_size))
            # return random.choice(np.arange(self.action_size))
            # max_action = np.argmax(action_values.cpu().data.numpy())
            # action = action_list[max_action]
            return max_action#action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        ### Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)




def dqn(n_episodes=1_000_000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    episode = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        # import random
        action = random.randint(0,24)
        Discrete_to_continous_action = action_list[action]
        Discrete_to_continous_action = torch.from_numpy(Discrete_to_continous_action).to(device)
        # action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    
        score = 0
        for t in range(max_t):
            #env.render()
            state = np.atleast_2d(state)
            state = utility.normalised(state)

            state = torch.from_numpy(state)

            _, mu, log_var = vae(state.unsqueeze(0).to(device))
            sigma = log_var.exp()
            eps = torch.randn_like(sigma)
            z = eps.mul(sigma).add_(mu).squeeze(0)
            z = z.cpu().detach().numpy()
            z = np.atleast_2d(z)
            # z = utility.normalised(z)
            
            # print("dddddddddddddddddddddd",z)

            # action = agent.act(state, episode)
            # action = agent.act(z, episode)

            z = torch.from_numpy(z.squeeze(0)).to(device)



            rnn_input = torch.cat([z, Discrete_to_continous_action], dim=-1).float()

            _, hidden,_ = rnn(rnn_input.unsqueeze(0))

            Concatenated_current_state = torch.cat((z.to(device), hidden[0]),-1).cpu().detach().numpy()

            action = agent.act(Concatenated_current_state, episode)
            
            Discrete_to_contin_action = action_list[action]
            Discrete_to_continous_action = torch.from_numpy(Discrete_to_contin_action).to(device)


            # action = agent.act(state, episode)
            next_state, reward, done, _ = env.step(Discrete_to_continous_action.cpu())
            next_state_ = next_state

            next_state_ = np.atleast_2d(next_state_)
            next_state_ = utility.normalised(next_state_)
            next_state_ = torch.from_numpy(next_state_)

            _, mu, log_var = vae(next_state_.unsqueeze(0).to(device))
            sigma = log_var.exp()
            eps = torch.randn_like(sigma)
            next_state_ = eps.mul(sigma).add_(mu).squeeze(0)
            next_state_ = next_state_.cpu().detach().numpy()
            next_state_ = np.atleast_2d(next_state_)
            # next_state_ = utility.normalised(next_state_)
            next_state_ = torch.from_numpy(next_state_.squeeze(0)).to(device)
            
            Concatenated_next_state_ = torch.cat((next_state_.to(device), hidden[0]),-1).cpu().detach().numpy()
            # print("dddddddddddddddddddddd",z)

            # action = agent.act(state, episode)
            # action = agent.act(z, episode)

     

            agent.step(Concatenated_current_state, action, reward, Concatenated_next_state_, done)
            # print("state",state)
            # print("action",action)
            # print("reward",reward)
            # print("next_dddd",next_state)

            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        episode = max(eps_end, eps_decay*episode) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_.pth')
            break
    return scores

agent = Agent(state_size=287, action_size=25, seed=0)
scores = dqn()