import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, datetime
from os.path import join
import time
import numpy as np
from ENVIRONMENT.Socnavenv import SocNavEnv
import random
import os.path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
import argparse
time_steps = 200

parser = argparse.ArgumentParser("number episodes asigning")
parser.add_argument('--episodes', type=int,
                    help="Number of episodes.")

args = parser.parse_args()
rollout_dir = 'data/'
if not os.path.exists(rollout_dir):
    os.makedirs(rollout_dir)

total_episodes = args.episodes


def discrete_to_continuous_action(action:int):
    """
    Function to return a continuous space action for a given discrete action
    """
    if action == 0:
        return np.array([0, 0.125], dtype=np.float32)
    
    elif action == 1:
        return np.array([0, -0.125], dtype=np.float32)

    elif action == 2:
        return np.array([1, 0.125], dtype=np.float32) 
    
    elif action == 3:
        return np.array([1, -0.125], dtype=np.float32) 

    elif action == 4:
        return np.array([1, 0], dtype=np.float32)

    elif action == 5:
        return np.array([-1, 0], dtype=np.float32)
    
    else:
        raise NotImplementedError


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

        env = SocNavEnv(relative_observations=True)
        s = 0
        start_time = time.time()
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
                action_ = random.randint(0, 5)
                action = discrete_to_continuous_action(action_)

                nxt_obs, nxt_reward, done, _ = env.step(action)
                prev_action = action 
                action = torch.from_numpy(action).float()
                obs = torch.from_numpy(obs).float()
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
            torch.save(self.data_dic, self.dir_name + 'saved_vae_rollout_train.pt') 
        elif self.mode  == 'test':
            torch.save(self.data_dic, self.dir_name + 'saved_vae_rollout_test.pt')
        elif self.mode  == 'val':
            torch.save(self.data_dic, self.dir_name + 'saved_vae_rollout_validation.pt')
       
       
        
    # #     end_time = time.time()-start_time
    # #     times = str(datetime.timedelta(seconds=end_time)).split(".")
    # #     print('Finished in {0}'.format(times[0]))
        
    # def pad_tensor(self, tensor, pad):
    #     pad_size = pad - tensor.size(0)
    #     return torch.cat([tensor.to(device), torch.zeros([pad_size, tensor.size(1)]).to(device)], dim=0)


    
    # def fit_dataset_to_rnn(self):
        
    #     for episode_idx, episode_data in enumerate(self.data_dic.values()):

    #         obs_sequence = episode_data['obs_sequence']

    #         action_sequence = episode_data['action_sequence']
    #         reward_sequence = episode_data['reward_sequence']
    #         done_sequence = episode_data['done_sequence']

    #         obs_sequence = torch.stack(obs_sequence, dim=0).squeeze(1)
    #         obs_sequence = self.pad_tensor(obs_sequence, pad=time_steps).cpu().detach().numpy()
    #         obs_sequence = torch.from_numpy(obs_sequence) 
  
            
    #         done = [int(d) for d in done_sequence]
    #         done = torch.tensor(done).unsqueeze(-1)
    #         done_sequence = self.pad_tensor(done, pad=time_steps)
            
    #         action_sequence = torch.stack(action_sequence, dim=0).squeeze(1)
    #         action_sequence = self.pad_tensor(action_sequence, pad=time_steps)
 
            
    #         reward = torch.tensor(reward_sequence).unsqueeze(-1)
    #         reward_sequence = self.pad_tensor(reward, pad=time_steps)
    #         reward = self.pad_tensor(reward, pad=time_steps)
            
    #         episode_data['obs_sequence'] = obs_sequence
    #         episode_data['action_sequence'] = action_sequence
    #         episode_data['done_sequence'] = done_sequence
    #         episode_data['reward_sequence'] = reward_sequence

    #         if self.mode == 'train':
    #             torch.save(self.data_dic, self.dir_name + 'saved_rnn_rollout_train.pt') 
    #         elif self.mode  == 'test':
    #             torch.save(self.data_dic, self.dir_name + 'saved_rnn_rollout_test.pt')
    #         elif self.mode  == 'val':
    #             torch.save(self.data_dic, self.dir_name + 'saved_rnn_rollout_validation.pt')

   
        
       


rollout_dic = {}
rollout_dir = 'data/'
train_dataset = Rollout(rollout_dic, rollout_dir,'train', int(total_episodes))
train_dataset.make_rollout()
# train_dataset.fit_dataset_to_rnn()

rollout_dic = {}
rollout_dir = 'data/'
val_dataset = Rollout(rollout_dic, rollout_dir,'test', int(total_episodes*0.1))
val_dataset.make_rollout()
# val_dataset.fit_dataset_to_rnn()


rollout_dic = {}
rollout_dir = 'data/'
test_dataset = Rollout(rollout_dic, rollout_dir,'val', int(total_episodes*0.1))
test_dataset.make_rollout()
# test_dataset.fit_dataset_to_rnn()



