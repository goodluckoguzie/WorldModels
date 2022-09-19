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
time_steps = 350

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


def discrete_to_continuous_action(action:int):
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
    
    # elif action == 6:
    #     return np.array([-0.8, +0.4], dtype=np.float32)

    # elif action == 7:
    #     return np.array([-0.8, -0.4], dtype=np.float32)
    
    else:
        raise NotImplementedError

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
    return observation


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
                action_ = random.randint(0, 5)
                action = discrete_to_continuous_action(action_)
                obs = preprocess_observation(obs)

                nxt_obs, nxt_reward, done, _ = env.step(action)
                prev_action = action 
                # action = torch.from_numpy(action).float()
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
            torch.save(self.data_dic, self.dir_name + 'saved_vae_rollout_train.pt') 
        elif self.mode  == 'test':
            torch.save(self.data_dic, self.dir_name + 'saved_vae_rollout_test.pt')
        elif self.mode  == 'val':
            torch.save(self.data_dic, self.dir_name + 'saved_vae_rollout_validation.pt')



if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")


    env.configure('./configs/env.yaml')
    
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
