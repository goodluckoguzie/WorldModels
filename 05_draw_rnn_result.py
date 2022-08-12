import torch
from torch import nn
import os
import sys
from matplotlib.pyplot import axis
import random
import gym
import cv2
import numpy as np
from ENVIRONMENT import socnavenv
from ENVIRONMENT.socnavenv import SocNavEnv
from tqdm import tqdm
from UTILITY import utility
from UTILITY.utility import transform_processed_observation_into_raw
from RNN.RNN import RNN,Rnn
import argparse
parser = argparse.ArgumentParser("mode asigning")
parser.add_argument('--mode', type=str, required=True,help="normal,window,reward")
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = torch.load('./data/saved_rollout_rnn_test.pt')

latents = 31
actions = 2
hiddens = 256
batch_size = 1
timestep = 200
train_window = 10 # our sliding window value

def trains(mode='normal'):

    class MDN_Dataset(torch.utils.data.Dataset):
        def __init__(self, MDN_data):
            self.MDN_data = MDN_data
        def __len__(self):
            return len(self.MDN_data)

        def __getitem__(self, idx):
            data = self.MDN_data[idx]
            obs = data['obs_sequence']
            action = data['action_sequence']
            #reward = data['reward_sequence']
            return  (action, obs)

    test_dataset = MDN_Dataset(dataset)


    if mode == 'normal':

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        rnn = RNN(latents, actions, hiddens).to(device)
        rnn.load_state_dict(torch.load("./MODEL/MDN_RNN_window.pt"))

        rnn.eval()
        for batch_idx, (action, obs) in enumerate(test_dataloader):
            current_timestep = obs[:, :-1, :].to("cuda:0") # remove the last timestep
            action = action[:, :-1, :].to("cuda:0")  # remove the last action and timestep
            nxt_timestep = obs[:, 1:, :].to("cuda:0")  #remove the first timestep and include the last timestep

            states = torch.cat([current_timestep, action], dim=-1)
            predicted_nxt_timestep, _, _ = rnn(states)

            nxt_timestep = nxt_timestep
            predicted_nxt_timestep = predicted_nxt_timestep

            steps = predicted_nxt_timestep.shape[0]

            for step in range(steps):
                current_timestep = current_timestep[0, step, :]
                current_timestep = current_timestep.cpu().detach().numpy()
            
                nxt_timestep = nxt_timestep[0, step, :]
                nxt_timestep = nxt_timestep.cpu().detach().numpy()

                predicted_nxt_timestep = predicted_nxt_timestep[0, step, :]
                predicted_nxt_timestep = predicted_nxt_timestep.cpu().detach().numpy()

                # Check if all elements in array are zero
                result = np.all((nxt_timestep == 0))
                if result:
                    print('Array contains only 0')
                    continue
                else:
                        print('Array has non-zero items too')

                input_sample = np.atleast_2d(nxt_timestep)
                nxt_timestep = nxt_timestep.flatten()

                predicted_nxt_timestep = np.atleast_2d(predicted_nxt_timestep)
                predicted_nxt_timestep = predicted_nxt_timestep.flatten()


                robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(current_timestep)
                image0 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "current timestep", dont_draw=True)
                current_grey = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("next timestep", image1)
    
                robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(nxt_timestep)
                image1 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "next timestep", dont_draw=True)
                next_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("next timestep", image1)
    
                robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(predicted_nxt_timestep)
                image2 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "predicted next timestep", dont_draw=True)
                next_predicted_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("predicted next timestep", image2)

                merged = cv2.merge([current_grey, next_grey, next_predicted_grey])
                # print(merged.shape, merged.dtype)
                #merged = np.dstack((current_grey, next_grey, next_predicted_grey))
                print(merged.shape, merged.dtype)
                cv2.imshow("Merged", merged)



                for j in tqdm(range(100)):
                    k = cv2.waitKey(10)
                    if k%255 == 27:
                        sys.exit(0)



    elif mode == 'window':

        def create_inout_sequences(input_data,action_data, tw):
            inout_seq = []
            for i in range(timestep-tw):#the timestep is gotten from the extracted data
                train_seq = input_data[:,i:i+tw,:] #get the 1 to 10 , label is the 11th - get 2 to 11 label 12th .....
                train_label = input_data[:,i+tw:i+tw+1,:]

                action_seq = action_data[:,i:i+tw,:]
                action_label = action_data[:,i+tw:i+tw+1,:]
                inout_seq.append((train_seq ,train_label,action_seq,action_label))
            return inout_seq

        class MDN_Dataset(torch.utils.data.Dataset):
            def __init__(self, MDN_data):
                self.MDN_data = MDN_data
            def __len__(self):
                return len(self.MDN_data)

            def __getitem__(self, idx):
                data = self.MDN_data[idx]
                obs = data['obs_sequence']
                action = data['action_sequence']
                #reward = data['reward_sequence']
                return (action, obs)


        test_dataset = MDN_Dataset(dataset)
        train_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        rnn = RNN(latents, actions, hiddens).to(device)
        rnn.load_state_dict(torch.load("./MODEL/MDN_RNN_window.pt"))

        rnn.eval()
        for batch_idx, (action, obs) in enumerate(train_dataloader):

            print(batch_idx)
            print(obs.shape)

            train_inout_seq = create_inout_sequences(obs, action, train_window) #get the action data in batches along with the expected true value
            for current_timestep, nxt_timestep,action,_ in train_inout_seq:
        
                action = action.to("cuda:0")  
                current_timestep = current_timestep.to("cuda:0")  
                nxt_timestep = nxt_timestep.to("cuda:0")  
                states = torch.cat([current_timestep, action], dim=-1)

                predicted_nxt_timestep, _, _ = rnn(states)
                current_timestep = current_timestep
                nxt_timestep = nxt_timestep
                predicted_nxt_timestep = predicted_nxt_timestep[:, -1:, :]


                steps = predicted_nxt_timestep.shape[0]
        
                for step in range(steps):
                    current_timestep = current_timestep[0, train_window-1, :]
                    current_timestep = current_timestep.cpu().detach().numpy()

                    nxt_timestep = nxt_timestep[0, step, :]
                    nxt_timestep = nxt_timestep.cpu().detach().numpy()

                    predicted_nxt_timestep = predicted_nxt_timestep[0, step, :]
                    predicted_nxt_timestep = predicted_nxt_timestep.cpu().detach().numpy()


                    print("output_sample")
                    print(predicted_nxt_timestep)
                    print("input_sample")
                    print(current_timestep)
                    print("")
                    # Check if all elements in array are zero. this help us to remove the padded timestep
                    result = np.all((nxt_timestep == 0))
                    if result:
                        print('Array contains only 0')
                        continue
                    else:
                        print('Array has non-zero items too')

                    nxt_timestep = np.atleast_2d(nxt_timestep)
                    #input_sample = utility.denormalised(input_sample)
                    nxt_timestep = nxt_timestep.flatten()

                    predicted_nxt_timestep = np.atleast_2d(predicted_nxt_timestep)
                    #utility.denormalised(output_sample)
                    predicted_nxt_timestep = predicted_nxt_timestep.flatten()

                    print("output_sample")
                    print(predicted_nxt_timestep)
                    print("input_sample")
                    print(current_timestep)
                    print("")

                    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(current_timestep)
                    image0 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "current timestep", dont_draw=True)
                    current_grey = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("next timestep", image1)
        
                    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(nxt_timestep)
                    image1 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "next timestep", dont_draw=True)
                    next_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("next timestep", image1)
        
                    robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(predicted_nxt_timestep)
                    image2 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "predicted next timestep", dont_draw=True)
                    next_predicted_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("predicted next timestep", image2)

                    merged = cv2.merge([current_grey, next_grey, next_predicted_grey])
                    # print(merged.shape, merged.dtype)
                    #merged = np.dstack((current_grey, next_grey, next_predicted_grey))
                    print(merged.shape, merged.dtype)
                    cv2.imshow("Merged", merged)


                    for j in tqdm(range(100)):
                        k = cv2.waitKey(10)
                        if k%255 == 27:
                            sys.exit(0)



    elif mode == 'reward':

        def create_inout_sequences(input_data,action_data, reward_data,tw):
            inout_seq = []
            for i in range(timestep-tw):#the timestep is gotten from the extracted data this must tally with that of extract_data_for_rnn.py
                train_seq = input_data[:,i:i+tw,:]
                train_label = input_data[:,i+tw:i+tw+1,:]

                action_seq = action_data[:,i:i+tw,:]
                action_label = action_data[:,i+tw:i+tw+1,:]

                reward_seq = reward_data[:,i:i+tw,:]
                reward_label = reward_data[:,i+tw:i+tw+1,:]
                inout_seq.append((train_seq ,train_label,action_seq,action_label,reward_seq,reward_label))
            return inout_seq

        class MDN_Dataset(torch.utils.data.Dataset):
            def __init__(self, MDN_data):
                self.MDN_data = MDN_data
            def __len__(self):
                return len(self.MDN_data)

            def __getitem__(self, idx):
                data = self.MDN_data[idx]
                obs = data['obs_sequence']
                action = data['action_sequence']
                reward = data['reward_sequence']
                return (action, obs,reward)

        test_dataset = MDN_Dataset(dataset)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        reward = 1
        rnn = Rnn(latents, actions,reward, hiddens).to(device)
        rnn.load_state_dict(torch.load("./model/MDN_RNN_reward.pt"))
  
        rnn.eval()
        for batch_idx, (action, obs,reward) in enumerate(test_dataloader):# get a batch of timesteps seperated by episodes
            print("batch_idx")
            print(batch_idx)

            train_inout_seq = create_inout_sequences(obs, action,reward, train_window) #using the a sliding window of 10 . the the first 10 time step and the 11th timetep will be our label.
                                                                        # next shift the sliding window a step ahead now our label is the 12th timestep

            w = 0
            for current_timestep, nxt_timestep,action,_ ,reward, nxt_reward in train_inout_seq:
                action = action.to("cuda:0")  
                current_timestep = current_timestep.to("cuda:0")  
                nxt_timestep = nxt_timestep.to("cuda:0")
                reward = reward.type(torch.cuda.FloatTensor)
                states = torch.cat([current_timestep, action,reward], dim=-1)

                predicted_nxt_timestep, _, _ = rnn(states)
                nxt_timestep = nxt_timestep
                predicted_nxt_timestep = predicted_nxt_timestep[:, -1:, :]

                steps = predicted_nxt_timestep.shape[0]
                for step in range(steps):
                    current_timestep = current_timestep[0, train_window-1, :]
                    current_timestep = current_timestep.cpu().detach().numpy()
                                   
                    nxt_timestep = nxt_timestep[0, step, :]
                    nxt_timestep = nxt_timestep.cpu().detach().numpy()

                    predicted_nxt_timestep = predicted_nxt_timestep[0, step, :]
                    predicted_nxt_timestep = predicted_nxt_timestep.cpu().detach().numpy()

                    # Check if all elements in array are zero. this help us to remove the padded timestep
                    result = np.all((nxt_timestep == 0))
                    if result:
                        print('Array contains only 0')
                        continue
                    else:
                        print('Array has non-zero items too')

                    nxt_timestep = np.atleast_2d(nxt_timestep)
                    #input_sample = utility.denormalised(input_sample)
                    nxt_timestep = nxt_timestep.flatten()

                    predicted_nxt_timestep = np.atleast_2d(predicted_nxt_timestep)
                    #utility.denormalised(output_sample)
                    predicted_nxt_timestep = predicted_nxt_timestep.flatten()

                    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(current_timestep)
                    image0 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "current timestep", dont_draw=True)
                    current_grey = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("next timestep", image1)
        
                    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(nxt_timestep)
                    image1 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "next timestep", dont_draw=True)
                    next_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("next timestep", image1)
        
                    robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(predicted_nxt_timestep)
                    image2 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "predicted next timestep", dont_draw=True)
                    next_predicted_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("predicted next timestep", image2)

                    merged = cv2.merge([current_grey, next_grey, next_predicted_grey])
                    # print(merged.shape, merged.dtype)
                    #merged = np.dstack((current_grey, next_grey, next_predicted_grey))
                    print(merged.shape, merged.dtype)
                    cv2.imshow("Merged", merged)


                    for j in tqdm(range(100)):
                        k = cv2.waitKey(10)
                        if k%255 == 27:
                            sys.exit(0)


    elif mode == 'dream':

        def create_inout_sequences(input_data,action_data, tw):
            inout_seq = []
            for i in range(timestep-tw):#the timestep is gotten from the extracted data
                train_seq = input_data[:,i:i+tw,:]
                train_label = input_data[:,i+tw:i+tw+1,:]

                action_seq = action_data[:,i:i+tw,:]
                action_label = action_data[:,i+tw:i+tw+1,:]
                inout_seq.append((train_seq ,train_label,action_seq,action_label))
            return inout_seq

        class MDN_Dataset(torch.utils.data.Dataset):
            def __init__(self, MDN_data):
                self.MDN_data = MDN_data
            def __len__(self):
                return len(self.MDN_data)

            def __getitem__(self, idx):
                data = self.MDN_data[idx]
                obs = data['obs_sequence']
                action = data['action_sequence']
                #reward = data['reward_sequence']
                return (action, obs)

        test_dataset = MDN_Dataset(dataset)
        train_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        rnn = RNN(latents, actions, hiddens).to(device)
        rnn.load_state_dict(torch.load("./MODEL/MDN_RNN_window.pt"))

        rnn.eval()
        for batch_idx, (action, obs) in enumerate(train_dataloader):
            print(batch_idx)
            i = 0
            train_inout_seq = create_inout_sequences(obs, action, train_window) #get the action data in bataches along with the expected true value
            for current_timestep, nxt_timestep,action,_ in train_inout_seq:
                print("i")
                print(i)
        
                if i > 0:# ignore dream state only for the first time step
                    print("dreaming")
                    current_timestep = current_timestep.to("cuda:0")

                    dream_input = output_sample_.to("cuda:0")
                    current_timestep[:, -1:, :] = dream_input[:, -1:, :]#replace the 10th timestep with the predicted timestep
                    dream_current_timestep = current_timestep
                    action = action.to("cuda:0")
                    
                    nxt_timestep = nxt_timestep.to("cuda:0")  
                    states = torch.cat([dream_current_timestep, action], dim=-1)
                    
                    predicted_nxt_timestep, _, _ = rnn(states)
                    output_sample = predicted_nxt_timestep

                    output_sample = output_sample[:, -1:, :]#the predicted state 
                    input_sample = nxt_timestep#the action next state 
                    output_sample_ = predicted_nxt_timestep.to("cuda:0") 
                else :
                    action = action.to("cuda:0")  # get the action for first 10 timesteps for the episoed (total of 200 timesteps)
                    current_timestep = current_timestep.to("cuda:0")  # get the observations for first 10 timesteps for the episoed (total of 200 timesteps)
                    nxt_timestep = nxt_timestep.to("cuda:0")  #get the only the next timestep{1,2,3,4,5,6,7,8,9,10} = label for this will be 11th timestep
                    states = torch.cat([current_timestep, action], dim=-1)

                    predicted_nxt_timestep, _, _ = rnn(states)
                    output_sample_ = predicted_nxt_timestep.to("cuda:0") #this will be use from the dreaming mode
                    output_sample = predicted_nxt_timestep[:, -1:, :].to("cuda:0") 
                i = 1+i

                steps = output_sample.shape[0]

                for step in range(steps):
                    print("step")
                    print(step)
                
                    nxt_timestep = output_sample[0, step, :]
                    nxt_timestep = nxt_timestep.cpu().detach().numpy()


                    predicted_nxt_timestep = predicted_nxt_timestep[0, step, :]
                    predicted_nxt_timestep = predicted_nxt_timestep.cpu().detach().numpy()
                
                    # Check if all elements in array are zero
                    result = np.all((current_timestep == 0))
                    if result:
                        print('Array contains only 0')
                        continue
                    else:
                        print('Array has non-zero items too')

                        nxt_timestep = np.atleast_2d(nxt_timestep)
                        #input_sample = utility.denormalised(input_sample)
                        nxt_timestep = nxt_timestep.flatten()

                        predicted_nxt_timestep = np.atleast_2d(predicted_nxt_timestep)
                        #utility.denormalised(output_sample)
                        predicted_nxt_timestep = predicted_nxt_timestep.flatten()


                    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(current_timestep)
                    image0 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "current timestep", dont_draw=True)
                    current_grey = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("next timestep", image1)
        
                    robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(nxt_timestep)
                    image1 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "next timestep", dont_draw=True)
                    next_grey = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("next timestep", image1)
        
                    robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(predicted_nxt_timestep)
                    image2 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "predicted next timestep", dont_draw=True)
                    next_predicted_grey = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("predicted next timestep", image2)

                    merged = cv2.merge([current_grey, next_grey, next_predicted_grey])
                    # print(merged.shape, merged.dtype)
                    #merged = np.dstack((current_grey, next_grey, next_predicted_grey))
                    print(merged.shape, merged.dtype)
                    cv2.imshow("Merged", merged)


                    for j in tqdm(range(100)):
                        k = cv2.waitKey(10)
                        if k%255 == 27:
                            sys.exit(0)
            
trains(args.mode)