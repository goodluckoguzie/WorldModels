import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, datetime
from os.path import join
import time
import numpy as np
from ENVIRONMENT.socnavenv import SocNavEnv
import random
import os.path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
import argparse
time_steps = 50



parser = argparse.ArgumentParser("total_episodes asigning")
parser.add_argument('--episodes', type=int,
                    help="Number of episodes.")


parser.add_argument('--testepisodes', type=int,
                    help="Number of episodes.") 
args = parser.parse_args()
rollout_dir = 'data/'
if not os.path.exists(rollout_dir):
    os.makedirs(rollout_dir)

total_episodes = args.episodes






class Test_Rollout():
    def __init__(self, data_dic, dir_name):
        super().__init__()
        self.data_dic = data_dic
        self.dir_name = dir_name
        
    def make_rollout_test(self):
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

        env = SocNavEnv()
        s = 0
        start_time = time.time()
        total_episodes = args.testepisodes
        while s < total_episodes:
            obs_sequence = []
            action_sequence = []
            reward_sequence= []
            done_sequence = []
            obs = env.reset()
            
            prev_action = None
            for t in range(time_steps):
                #env.render()
                action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                obs, reward, done, _ = env.step(action)
                prev_action = action 
                action = torch.from_numpy(action).float()
                obs = torch.from_numpy(obs).float()
                obs_sequence.append(obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
        
                t+=1
                if done:
        
                    print("Episode [{}/{}] finished after {} timesteps".format(s + 1,total_episodes, t), flush=True)
                    obs = env.reset()
                    break
            self.data_dic[s] = {"obs_sequence":obs_sequence, "action_sequence":action_sequence, 
                        "reward_sequence":reward_sequence, "done_sequence":done_sequence}        
            s+=1
        torch.save(self.data_dic, self.dir_name  + "saved_rollout_test.pt")
       
        
        end_time = time.time()-start_time
        times = str(datetime.timedelta(seconds=end_time)).split(".")
        print('Finished in {0}'.format(times[0]))
        
    def pad_tensor(self, tensor, pad):
        pad_size = pad - tensor.size(0)
        return torch.cat([tensor.to(device), torch.zeros([pad_size, tensor.size(1)]).to(device)], dim=0)


    
    def fit_dataset_to_test_rnn(self):
        
        for episode_idx, episode_data in enumerate(self.data_dic.values()):

            obs_sequence = episode_data['obs_sequence']

            action_sequence = episode_data['action_sequence']
            reward_sequence = episode_data['reward_sequence']
            done_sequence = episode_data['done_sequence']

            obs_sequence = torch.stack(obs_sequence, dim=0).squeeze(1)
            obs_sequence = self.pad_tensor(obs_sequence, pad=time_steps).cpu().detach().numpy()
            obs_sequence = torch.from_numpy(obs_sequence) 
  
            
            done = [int(d) for d in done_sequence]
            done = torch.tensor(done).unsqueeze(-1)
            done_sequence = self.pad_tensor(done, pad=time_steps)
            
            action_sequence = torch.stack(action_sequence, dim=0).squeeze(1)
            action_sequence = self.pad_tensor(action_sequence, pad=time_steps)
 
            
            reward = torch.tensor(reward_sequence).unsqueeze(-1)
            reward_sequence = self.pad_tensor(reward, pad=time_steps)
            reward = self.pad_tensor(reward, pad=time_steps)
            
            episode_data['obs_sequence'] = obs_sequence
            episode_data['action_sequence'] = action_sequence
            episode_data['done_sequence'] = done_sequence
            episode_data['reward_sequence'] = reward_sequence
        torch.save(self.data_dic, self.dir_name + 'saved_rollout_rnn_test.pt')           
        
############################################################################################

class Train_Rollout():
    def __init__(self, data_dic, dir_name):
        super().__init__()
        self.data_dic = data_dic
        self.dir_name = dir_name
        
    def make_rollout(self):
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)

        env = SocNavEnv()
        s = 0
        start_time = time.time()
        while s < total_episodes:
            obs_sequence = []
            action_sequence = []
            reward_sequence= []
            done_sequence = []
            obs = env.reset()
            
            prev_action = None
            for t in range(time_steps):
                #env.render()
                action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
                obs, reward, done, _ = env.step(action)
                prev_action = action 
                action = torch.from_numpy(action).float()
                obs = torch.from_numpy(obs).float()
                obs_sequence.append(obs)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
        
                t+=1
                if done:
        
                    print("Episode [{}/{}] finished after {} timesteps".format(s + 1,total_episodes, t), flush=True)
                    obs = env.reset()
                    break
            self.data_dic[s] = {"obs_sequence":obs_sequence, "action_sequence":action_sequence, 
                        "reward_sequence":reward_sequence, "done_sequence":done_sequence}        
            s+=1
        torch.save(self.data_dic, self.dir_name  + "saved_rollout_train.pt")
       
        
        end_time = time.time()-start_time
        times = str(datetime.timedelta(seconds=end_time)).split(".")
        print('Finished in {0}'.format(times[0]))
        
    def pad_tensor(self, tensor, pad):
        pad_size = pad - tensor.size(0)
        return torch.cat([tensor.to(device), torch.zeros([pad_size, tensor.size(1)]).to(device)], dim=0)


    
    def fit_dataset_to_rnn(self):
        
        for episode_idx, episode_data in enumerate(self.data_dic.values()):

            obs_sequence = episode_data['obs_sequence']

            action_sequence = episode_data['action_sequence']
            reward_sequence = episode_data['reward_sequence']
            done_sequence = episode_data['done_sequence']

            obs_sequence = torch.stack(obs_sequence, dim=0).squeeze(1)
            obs_sequence = self.pad_tensor(obs_sequence, pad=time_steps).cpu().detach().numpy()
            obs_sequence = torch.from_numpy(obs_sequence) 
  
            
            done = [int(d) for d in done_sequence]
            done = torch.tensor(done).unsqueeze(-1)
            done_sequence = self.pad_tensor(done, pad=time_steps)
            
            action_sequence = torch.stack(action_sequence, dim=0).squeeze(1)
            action_sequence = self.pad_tensor(action_sequence, pad=time_steps)
 
            
            reward = torch.tensor(reward_sequence).unsqueeze(-1)
            reward_sequence = self.pad_tensor(reward, pad=time_steps)
            reward = self.pad_tensor(reward, pad=time_steps)
            
            episode_data['obs_sequence'] = obs_sequence
            episode_data['action_sequence'] = action_sequence
            episode_data['done_sequence'] = done_sequence
            episode_data['reward_sequence'] = reward_sequence
        torch.save(self.data_dic, self.dir_name + 'saved_rollout_rnn_train.pt')  

     
        



rollout_dic = {}
rollout_dir = 'data/'
ro = Train_Rollout(rollout_dic, rollout_dir)
ro.make_rollout()
latents = 31
ro.fit_dataset_to_rnn()

rollout_dic = {}
ro = Test_Rollout(rollout_dic, rollout_dir)
ro.make_rollout_test()
latents = 31
ro.fit_dataset_to_test_rnn()    