import numpy as np
import os, sys, glob
import gym
from hparams import WorldFrameHyperParams as hp
from hparams import WorldFrame_Datasets_Timestep_2 as data

import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
import os
import torch



def discrete_to_continuous_action(action:int):
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
    return torch.from_numpy(observation)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################################################Padding our Observation#############################################


# def pad_tensor( tensor, pad):
#     pad_size = pad - tensor.size(0)
#     # print("tensor.size(0)",tensor.size())
#     return torch.cat([tensor.to(device), torch.zeros([pad_size, tensor.size(1)]).to(device)], dim=0)
def pad_tensor(t, episode_length, window_length=9, pad_function=torch.zeros):
    pad_size = episode_length - t.size(0) + window_length
    # Add window lenght - 1 infront of the number of obersavtion
    begin_pad       = pad_function([window_length-1, t.size(1)]).to(device)
    # pad the environment with lenght of the episode subtracted from  the total episode length
    episode_end_pad = pad_function([pad_size,      t.size(1)]).to(device)

    return torch.cat([begin_pad,t.to(device),episode_end_pad], dim=0)

###########################################End#############################################################################

def rollout():
    time_steps = data.time_steps

    env = gym.make("SocNavEnv-v1")
    # env.configure('./configs/env.yaml')
    env.configure('./configs/env_timestep_2.yaml')

    env.set_padded_observations(True)
    env = WorldFrameObservations(env)
    # seq_len = 300
    max_ep = 5000# hp.n_rollout
    feat_dir = data.data_dir

    os.makedirs(feat_dir, exist_ok=True)

    for ep in range(max_ep):
        obs_lst, action_lst, reward_lst, next_obs_lst, done_lst = [], [], [], [], []
        obs = env.reset()

        obs = preprocess_observation(obs)   
        done = False
        t = 0
        for t in range(time_steps):       
            # env.render()

            action_ = np.random.randint(0, 4)
            action = discrete_to_continuous_action(action_)
            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess_observation(next_obs)
            action = torch.from_numpy(action)
            np.savez(
                os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep,t)),
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
            
            obs_lst.append(obs)

            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)
            obs = next_obs
            if done:
                print("Episode [{}/{}] finished after {} timesteps".format(ep + 1, max_ep, t), flush=True)
                obs = env.reset()
                obs_lst = torch.stack(obs_lst, dim=0).squeeze(1)
                # print("9999999999999999999999999999999999999999999999999999999999999999999999999", obs_lst.shape )

                # obs_lst = pad_tensor(obs_lst, episode_length=time_steps).cpu().detach().numpy()
                # obs_lst = torch.from_numpy(obs_lst) 
                next_obs_lst = torch.stack(next_obs_lst, dim=0).squeeze(1)
                # next_obs_lst = pad_tensor(next_obs_lst, episode_length=time_steps).cpu().detach().numpy()
                # next_obs_lst = torch.from_numpy(next_obs_lst) 

                done_lst = [int(d) for d in done_lst]
                done_lst = torch.tensor(done_lst).unsqueeze(-1)
                done_lst = pad_tensor(done_lst, episode_length=time_steps).cpu().detach().numpy()
                done_lst=torch.from_numpy(done_lst)
                
                action_lst = torch.stack(action_lst, dim=0).squeeze(1)
                # action_lst = pad_tensor(action_lst, episode_length=time_steps).cpu().detach().numpy()
                # action_lst=torch.from_numpy(action_lst)
                
                reward_lst = torch.tensor(reward_lst).unsqueeze(-1)
                reward_lst = pad_tensor(reward_lst, episode_length=time_steps).cpu().detach().numpy()
                reward_lst=torch.from_numpy(reward_lst)
                break

        
        np.savez(
            os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
            obs=np.stack(obs_lst, axis=0), # (T, C, H, W)
            action=np.stack(action_lst, axis=0), # (T, a)
            reward=np.stack(reward_lst, axis=0), # (T, 1)
            next_obs=np.stack(next_obs_lst, axis=0), # (T, C, H, W)
            done=np.stack(done_lst, axis=0), # (T, 1)
        )
        
        

if __name__ == '__main__':


    np.random.seed(hp.seed)
    rollout()
