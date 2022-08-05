
import torch
import torch.nn as nn
from RNN import RNN
import os, time, sys,gym,cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import socnavenv
from socnavenv import SocNavEnv
from tqdm import tqdm
from utility import transform_processed_observation_into_raw
import utility


device = 'cuda' if torch.cuda.is_available() else 'cpu'

latents = 31
actions = 2
hiddens = 256
timestep = 200

x = torch.load('./data/saved1_rollout_rnn.pt')


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

train_dataset = MDN_Dataset(x)


batch_size = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

rnn = RNN(latents, actions, hiddens).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
rnn.load_state_dict(torch.load("./model/MDN_RNN_slide.pt"))

train_window = 10 # 

rnn.eval()
for batch_idx, (action, obs) in enumerate(train_dataloader):
    print(batch_idx)
    i = 0
    train_inout_seq = create_inout_sequences(obs, action, train_window) #get the action data in bataches along with the expected true value
    for input, labels,action,_ in train_inout_seq:
        print("i")
        print(i)
 
        if i > 0:# ignore dream state only for the first time step
            print("dreaming")
            input = input.to("cuda:0")

            dream_input = output_sample_.to("cuda:0")
            input[:, -1:, :] = dream_input[:, -1:, :]#replace the 10th timestep with the predicted timestep
            dream_input = input
            action = action.to("cuda:0")
            
            labels = labels.to("cuda:0")  
            states = torch.cat([dream_input, action], dim=-1)
            
            predicted_Nxt_states, _, _ = rnn(states)
            output_sample = predicted_Nxt_states

            output_sample = output_sample[:, -1:, :]#the predicted state 
            input_sample = labels#the action next state 
            output_sample_ = predicted_Nxt_states.to("cuda:0") 
        else :
            action = action.to("cuda:0")  # get the action for first 10 timesteps for the episoed (total of 200 timesteps)
            input = input.to("cuda:0")  # get the observations for first 10 timesteps for the episoed (total of 200 timesteps)
            labels = labels.to("cuda:0")  #get the only the next timestep{1,2,3,4,5,6,7,8,9,10} = label for this will be 11th timestep
            states = torch.cat([input, action], dim=-1)

            predicted_Nxt_states, _, _ = rnn(states)
            input_sample = labels
            output_sample_ = predicted_Nxt_states.to("cuda:0") #this will be use from the dreaming mode
            output_sample = predicted_Nxt_states[:, -1:, :].to("cuda:0") 
        i = 1+i

        steps = output_sample.shape[0]

        for step in range(steps):
            print("step")
            print(step)
        
            input_sample = input_sample[0, step, :]
            input_sample = input_sample.cpu().detach().numpy()


            output_sample = output_sample[0, step, :]
            output_sample = output_sample.cpu().detach().numpy()
           
            # Check if all elements in array are zero
            result = np.all((input_sample == 0))
            if result:
                print('Array contains only 0')
                continue
            else:
                print('Array has non-zero items too')


                print("outputoeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                print(output_sample)
                print("input_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeele")
                print(input_sample)
                print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")

                input_sample = np.atleast_2d(input_sample)
                #input_sample = utility.denormalised(input_sample)
                input_sample = input_sample.flatten()

                output_sample = np.atleast_2d(output_sample)
                #utility.denormalised(output_sample)
                output_sample = output_sample.flatten()


                robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(input_sample)
                image1 = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, " next timestep", dont_draw=True)
                cv2.imshow(" next timestep", image1)
                (B, _, R) = cv2.split(image1)

                robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(output_sample)
                image2 = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "predicted next timestep", dont_draw=True)
                (_, G, _) = cv2.split(image2)
                cv2.imshow("predicted next timestep", image2)

                merged = cv2.merge([B, G,R])
                cv2.imshow("Merged", merged)

                for j in tqdm(range(100)):
                    k = cv2.waitKey(10)
                    if k%255 == 27:
                        sys.exit(0)
    



































"""
#def draw(mode='real'):
    #print("mode")
    #print(mode)
    for batch_idx, (action, obs) in enumerate(train_dataloader):
        print(batch_idx)
        print(obs.shape)

        train_inout_seq = create_inout_sequences(obs, action, train_window) #get the action data in bataches along with the expected true value
        for input, labels,action,_ in train_inout_seq:

            i = 0
            if mode == 'real':
                
                action = action.to("cuda:0")  
                input = input.to("cuda:0")  
                labels = labels.to("cuda:0")  
                states = torch.cat([input, action], dim=-1)

                predicted_Nxt_states, _, _ = rnn(states)
                input_sample = labels
                output_sample = predicted_Nxt_states

                

            elif mode == 'dream':
                if i > 0:
                    print("dreaming")

                    input = predicted_Nxt_states
                    action = action.to("cuda:0")

                    labels = labels.to("cuda:0")  
                    states = torch.cat([input, action], dim=-1)

                    predicted_Nxt_states, _, _ = rnn(states)
                    input_sample = labels
                    output_sample = predicted_Nxt_states
                else :
                    action = action.to("cuda:0")  
                    input = input.to("cuda:0")  
                    labels = labels.to("cuda:0")  
                    states = torch.cat([input, action], dim=-1)

                    predicted_Nxt_states, _, _ = rnn(states)
                    input_sample = labels
                    output_sample = predicted_Nxt_states
            

                steps = output_sample.shape[0]

                for step in range(steps):
                    print("step")
                    print(step)
                
                    input_sample = input_sample[0, step, :]
                    input_sample = input_sample.cpu().detach().numpy()


                    output_sample = output_sample[0, step, :]
                    output_sample = output_sample.cpu().detach().numpy()
                    # Check if all elements in array are zero
                    result = np.all((input_sample == 0))
                    if result:
                        print('Array contains only 0')
                        continue
                    else:
                        print('Array has non-zero items too')


                        print("outputoeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                        print(output_sample)
                        print("input_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeele")
                        print(input_sample)
               for batch_idx, (action, obs) in enumerate(train_dataloader):
        print(batch_idx)
        print(obs.shape)

        train_inout_seq = create_inout_sequences(obs, action, train_window) #get the action data in bataches along with the expected true value
        for input, labels,action,_ in train_inout_seq:

            i = 0
            if mode == 'real':
                
                action = action.to("cuda:0")  
                input = input.to("cuda:0")  
                labels = labels.to("cuda:0")  
                states = torch.cat([input, action], dim=-1)

                predicted_Nxt_states, _, _ = rnn(states)
                input_sample = labels
                output_sample = predicted_Nxt_states

                

            elif mode == 'dream':
                if i > 0:
                    print("dreaming")

                    input = predicted_Nxt_states
                    action = action.to("cuda:0")

                    labels = labels.to("cuda:0")  
                    states = torch.cat([input, action], dim=-1)

                    predicted_Nxt_states, _, _ = rnn(states)
                    input_sample = labels
                    output_sample = predicted_Nxt_states
                else :
                    action = action.to("cuda:0")  
                    input = input.to("cuda:0")  
                    labels = labels.to("cuda:0")  
                    states = torch.cat([input, action], dim=-1)

                    predicted_Nxt_states, _, _ = rnn(states)
                    input_sample = labels
                    output_sample = predicted_Nxt_states
            

                steps = output_sample.shape[0]

                for step in range(steps):
                    print("step")
                    print(step)
                
                    input_sample = input_sample[0, step, :]
                    input_sample = input_sample.cpu().detach().numpy()


                    output_sample = output_sample[0, step, :]
                    output_sample = output_sample.cpu().detach().numpy()
                    # Check if all elements in array are zero
                    result = np.all((input_sample == 0))
                    if result:
                        print('Array contains only 0')
                        continue
                    else:
                        print('Array has non-zero items too')


                        print("outputoeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                        print(output_sample)
                        print("input_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeele")
                        print(input_sample)
                        print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")

                        input_sample = np.atleast_2d(input_sample)
                        #input_sample = utility.denormalised(input_sample)
                        input_sample = input_sample.flatten()

                        output_sample = np.atleast_2d(output_sample)
                        #utility.denormalised(output_sample)
                        output_sample = output_sample.flatten()


                        robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(input_sample)
                        image = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "input", dont_draw=True)
                        cv2.imshow("input", image)



                        robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(output_sample)
                        image = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "output", dont_draw=True)
                        cv2.imshow("output", image)


                        for j in tqdm(range(100)):
                            k = cv2.waitKey(10)
                            if k%255 == 27:
                                sys.exit(0)
        i = 1 + i         output_sample = np.atleast_2d(output_sample)
                        #utility.denormalised(output_sample)
                        output_sample = output_sample.flatten()


                        robot_obs, goal_obs, humans_obs = transform_processed_observation_into_raw(input_sample)
                        image = SocNavEnv.render_obs(robot_obs, goal_obs, humans_obs, "input", dont_draw=True)
                        cv2.imshow("input", image)



                        robot_obs_o, goal_obs_o, humans_obs_o = transform_processed_observation_into_raw(output_sample)
                        image = SocNavEnv.render_obs(robot_obs_o, goal_obs_o, humans_obs_o, "output", dont_draw=True)
                        cv2.imshow("output", image)


                        for j in tqdm(range(100)):
                            k = cv2.waitKey(10)
                            if k%255 == 27:
                                sys.exit(0)
        i = 1 + i

draw('dream')
"""