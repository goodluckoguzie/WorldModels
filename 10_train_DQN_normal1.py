import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import itertools

from Socnavenv import DiscreteSocNavEnv

GAMMA = 0.95
BATCH_SIZE=512
BUFFER_SIZE=1_000_000_000
MIN_REPLAY_SIZE=5_000_000
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY=10_000
TARGET_UPDATE_FREQ=21_000

loss_fn = nn.MSELoss()
# class Network(nn.Module):
#     """
#     Define a politica para tomar uma ação a partir de um estado
#     """
#     def __init__(self,env):
#         super(Network, self).__init__()
#         super().__init__()
#         self.num_actions = env.action_space.n
#         self.state_dim = env.observation_space.shape[0]    
#         self.fc1 = torch.nn.Linear(self.state_dim,64,'linear')
#         self.hidden1 = nn.Dropout(0.1)
#         self.fc2 = torch.nn.Linear(64,64,'linear')
#         self.hidden2 = nn.Dropout(0.08)
#         self.fc3 = torch.nn.Linear(64,64,'linear')
#         self.hidden3 = nn.Dropout(0.05)
#         self.fc4 = torch.nn.Linear(64,64,'linear')
#         self.fc5 = torch.nn.Linear(64,self.num_actions,'linear')
        
#     def forward(self,x):
        
#         x = self.hidden1(self.fc1(x))
#         x = self.hidden2(F.relu(self.fc2(x)))
#         x = self.hidden3(F.relu(self.fc3(x)))        
#         x = F.relu(self.fc4(x))
#         y = F.relu(self.fc5(x))
#         return y
    


class Network(nn.Module):
    def __init__(self,env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))
        self.net = nn.Sequential(
            nn.Linear(in_features,64),
            nn.Tanh(),
            nn.Linear(64,env.action_space.n)
            )
    def forward(self,x):
        return self.net(x)

    def act(self,obs):
        obs_t = torch.as_tensor(obs,dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        #get action with highest q value
        max_q_index = torch.argmax(q_values, dim=1)[0]
        #turn the tensor to integer
        action = max_q_index.detach().item()
        return action

# env = gym.make('CartPole-v0')
# env = gym.make("MountainCar-v0")
env = DiscreteSocNavEnv()

replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0.0], maxlen=100)
episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(),lr=5e-4)
# initialize the replace buffer

obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, rew, done, _ = env.step(action)

    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

# #main training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START,EPSILON_END])

    rnd_sample = random.random()

    if rnd_sample <=epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs,rew,done,_ = env.step(action)
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward = episode_reward +  rew

    if done:
        obs = env.reset()

        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    #After solve,watch it play
    if len(rew_buffer) >= 100:
        if np.mean(rew_buffer) >= 3:
            while True:
                action = online_net.act(obs)
                obs,_,_,done = env.step(action)
                env.render()
                if done:
                    env.reset() 

    #start Gradient Step
    transition = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transition])
    actions = np.asarray([t[1] for t in transition])
    rews = np.asarray([t[2] for t in transition])
    dones = np.asarray([t[3] for t in transition])
    new_obses = np.asarray([t[4] for t in transition])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)


    #compute Target
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1-dones_t) * max_target_q_values

    #compute Loss
    q_values = online_net(obses_t)

    action_q_values = torch.gather(input=q_values, dim=1,index=actions_t)

    # loss = nn.functional.smooth_l1_loss(action_q_values, targets)
    loss = loss_fn(action_q_values, targets)


    #gradient descent

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #Update Taregt Network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())
        torch.save(online_net.state_dict(), 'net.pth')

    #logging
    if step % 100 == 0:
        print()
        print('Step', step)
        print('Avg Reward',np.mean(rew_buffer))



