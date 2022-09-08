import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io

import numpy as np
from collections import deque, namedtuple
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import random
from collections import deque
# For visualization
from gym.wrappers.monitoring import video_recorder
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# advance_split = 5
# rotation_split = 5

# advance_grid, rotation_grid = np.meshgrid(
#     np.linspace(-1, 1, advance_split),
#     np.linspace(-1, 1, rotation_split))

# action_list = np.hstack((
#     advance_grid.reshape(( advance_split*rotation_split, 1)),
#     rotation_grid.reshape((advance_split*rotation_split, 1))))
# number_of_actions = action_list.shape[0]

# from ENVIRONMENT.Socnavenv import SocNavEnv 
# env = SocNavEnv()
from ENVIRONMENT.Socnavenv import DiscreteSocNavEnv 

env = DiscreteSocNavEnv()
print('State shape: ', env.observation_space)
print('Number of actions: ', env.action_space)


class ReplayBuffer():
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

class DQNNet(nn.Module):
    def __init__(self, input, output):
        super(DQNNet, self).__init__()
        self.input = nn.Linear(input, 200)
        self.output = nn.Linear(200, output)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.output(x)
        return x
    
class DQN():
    def __init__(self, env, memory_size=10000, learning_rate=1e-3, batch_size=64, target_update=1000, gamma=0.95, eps=1, eps_min=0.1, eps_period=2000):
        super(DQN, self).__init__()
        self.env = env
                
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Deep Q network
        self.predict_net = DQNNet(env.observation_space.shape[0], env.action_space.n).to(self.device)
        # self.predict_net = DQNNet(env.observation_space.n, env.action_space.n).to(self.device)

        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Target network
        self.target_net = DQNNet(env.observation_space.shape[0], env.action_space.n).to(self.device)
        # self.target_net = DQNNet(env.observation_space.n, env.action_space.n).to(self.device)

        self.target_net.load_state_dict(self.predict_net.state_dict())
        self.target_update = target_update
        self.update_count = 0
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        
        # Learning setting
        self.gamma = gamma
        
        # Exploration setting
        self.eps = eps
        self.eps_min = eps_min
        self.eps_period = eps_period

    # Get the action
    def get_action(self, state):
        # Random action
        if np.random.rand() < self.eps:
            self.eps = self.eps - (1 - self.eps_min) / self.eps_period if self.eps > self.eps_min else self.eps_min
            return np.random.randint(0, self.env.action_space.n)
        
        # Get the action
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        q_values = self.predict_net(state).cpu().detach().numpy()
        action = np.argmax(q_values)
        
        return action
    
    # Learn the policy
    def learn(self):
        # Replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calculate values and target values
        target_values = (rewards + self.gamma * torch.max(self.target_net(next_states), 1)[0] * (1-dones)).view(-1, 1)
        predict_values = self.predict_net(states).gather(1, actions.view(-1, 1))
        
        # Calculate the loss and optimize the network
        loss = self.loss_fn(predict_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update the target network
        self.update_count += 1
        if self.update_count == self.target_update:
            self.target_net.load_state_dict(self.predict_net.state_dict())
            self.update_count = 0

def main():
    ep_rewards = deque(maxlen=100)
    # ENV_NAME = "Taxi-v3"
    # env = gym.make(ENV_NAME)
    env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")
    # env = env.unwrapped
    agent = DQN(env, batch_size=128, memory_size=100000, target_update=100, gamma=0.95, learning_rate=1e-4, eps_min=0.05, eps_period=5000)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        ep_reward = 0
        while True:
            action = agent.get_action(state)
            next_state, reward , done, _ = env.step(action)
       
            ep_reward += reward

            agent.replay_buffer.add(state, action, reward, next_state, done)
            agent.learn()
            
            if done:
                ep_rewards.append(ep_reward)
                if i % 100 == 0:
                    print("episode: {}\treward: {}\tepsilon: {}".format(i, round(np.mean(ep_rewards), 3), round(agent.eps, 3)))
                break

            state = next_state

if __name__ == '__main__':
    main()




