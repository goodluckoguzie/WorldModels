import torch
import torch.nn as nn
import numpy as np
import glob, os
from torchvision import transforms

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

class GameSceneDataset(torch.utils.data.Dataset):





    def __init__(self, data_path, training=True, test_ratio=0.01):
        self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_[0-9][0-9][0-9]_*.npz')))
        np.random.seed(0)
        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices)*(1.0-test_ratio))
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
        # self.test_indices = np.delete(indices, self.train_indices)
        self.indices = self.train_indices if training else self.test_indices
        # import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        npz = np.load(self.fpaths[self.indices[idx]])
        obs = npz['obs']
        # obs = transform(obs)
        # obs = obs.permute(2, 0, 1) # (N, C, H, W)
        return obs

    def __len__(self):
        return len(self.indices)



# class GameSceneDataset_new(torch.utils.data.Dataset):
#     def __init__(self, data_path, training=True, test_ratio=0.01):
#         self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_[0-9][0-9][0-9]_*.npz')))
#         np.random.seed(0)
#         indices = np.arange(0, len(self.fpaths))
#         n_trainset = int(len(indices)*(1.0-test_ratio))
#         self.train_indices = indices[:n_trainset]
#         self.test_indices = indices[n_trainset:]
#         # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
#         # self.test_indices = np.delete(indices, self.train_indices)
#         self.indices = self.train_indices if training else self.test_indices
#         # import pdb; pdb.set_trace()

#     def __getitem__(self, idx):
#         npz = np.load(self.fpaths[self.indices[idx]])
#         obs = npz['obs']
#         action = npz['action']
#         reward = npz['reward']  # (T, n_actions) np array

#         # obs = transform(obs)
#         # obs = obs.permute(2, 0, 1) # (N, C, H, W)
#         return obs,action,reward

#     def __len__(self):
#         return len(self.indices)


        

class GameEpisodeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, seq_len=20, seq_mode=True, training=True, test_ratio=0.01,episode_length=None):
        self.training = training
        self.episode_length = episode_length
        self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
        np.random.seed(0)
        # print("ddddddddddddddddddddddddddddffffffffffffffffffffffffffffffffffff",self.episode_length)

        indices = np.arange(0, len(self.fpaths))

        n_trainset = int(len(indices)*(1.0-test_ratio))

        self.train_indices = indices[:n_trainset]
        # print("train_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indic ",self.train_indices)

        self.test_indices = indices[n_trainset:]
        # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
        # self.test_indices = np.delete(indices, self.train_indices)
        self.indices = self.train_indices if training else self.test_indices
        self.seq_len = seq_len
        self.seq_mode = seq_mode
        # print("seq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_mode ",seq_len)

    def __getitem__(self, idx):


        def pad_tensor(t, episode_length=self.episode_length, window_length=self.seq_len-1, pad_function=torch.zeros):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            pad_size = episode_length - t.size(0) + window_length
            # Add window lenght - 1 infront of the number of obersavtion
            begin_pad       = pad_function([window_length-1, t.size(1)]).to(device)
            # pad the environment with lenght of the episode subtracted from  the total episode length
            episode_end_pad = pad_function([pad_size,      t.size(1)]).to(device)
            # print("paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",episode_end_pad.shape)

            return torch.cat([begin_pad,t.to(device),episode_end_pad], dim=0)



        npz = np.load(self.fpaths[self.indices[idx]])
        # print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr", idx )

        obs = npz['obs'] # (T, H, W, C) np array
        # print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111",obs.shape)

        obs = torch.from_numpy(obs) 
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",obs.shape)
        obs = pad_tensor(obs ,window_length=(self.seq_len-1)).cpu().detach().numpy()
        # obs = torch.from_numpy(obs) 


        actions = npz['action'] # (T, n_actions) np array
        # print("2222222222222222222222222222222222222222222222222222222222222222",actions.shape)

        actions = torch.from_numpy(actions) 

        actions = pad_tensor(actions, window_length=(self.seq_len-1)).cpu().detach().numpy()
        # actions=torch.from_numpy(actions)
        # print("8888888888888888888888888888888888888888888888888888888888888888888888888888888", obs )
        # print("9999999999999999999999999999999999999999999999999999999999999999999999999", obs.shape )

        T, C = obs.shape
        # T, H, W, C = obs.shape
        n_seq = T // self.seq_len
        end_seq = n_seq * self.seq_len # T' = end of sequence
        # print("0000000000000000000000000000000000000000000000000000000000000000000 ",n_seq )
        # print("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",self.seq_len )
        # print("cccccccccccccccccccccccccccccccccccccccccccccccccccc",obs.shape)
        # print("dssssssssssssssssssssssssssssssssssssssssssssssssssss",actions.shape)   
        #      
        # print("end_seqend_seqend_seqend_seqend_seqend_seqend_seq",end_seq)          
        obs = obs[:end_seq].reshape([-1, self.seq_len, C]) # (N_seq, seq_len, H, W, C)
        actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]]) # 

        # if args.seq_mode:
        #     start_range = max_len-self.seq_len
        #     for t in range(0, max_len-self.seq_len, self.seq_len):
        #         obs[t:t+self.seq_len]
        # else:
        #     rand_start = np.random.randint(max_len-self.seq_len)
        #     obs = obs[rand_start:rand_start+self.seq_len] # (T, H, W, C)
        #     actions = actions[rand_start:rand_start+self.seq_len]
        return obs, actions

    def __len__(self):
        return len(self.indices)

def collate_fn(data):
    # obs (B, N_seq, seq_len, H, W, C), actions (B, N_seq, seq_len, n_actions)
    obs, actions = zip(*data)
    obs, actions = np.array(obs), np.array(actions)
    # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh",obs)

    # print("ssssssssssssssssssssssssssssssssssssssssss",obs.shape)

    _,_, seq_len, C = obs.shape

    obs = obs.reshape([-1, C]) # (B*N_seq*seq_len, H, W, C)
    actions = actions.reshape([-1, seq_len, actions.shape[-1]]) # (B*n_seq, n_actions)
    obs_lst = []
    for i in range(len(obs)): # batch loop
        # print(obs[i])
        obs_lst.append(torch.from_numpy(obs[i]))
        # obs_lst.append(transform(obs[i]))
        # for j in range(len(obs[i])): # sequence loop
        #     obs_lst.append(transform(obs[i][j]))
    obs = torch.stack(obs_lst, dim=0) # (B*N_seq*seq_len, C, H, W)
    # obs = obs.view([-1, seq_len, H, W, C]) # (B*N_seq, seq_len, C, H, W)
    return obs, torch.tensor(actions, dtype=torch.float)


class GameEpisodeDatasetNonPrePadded(torch.utils.data.Dataset):

    def __init__(self, data_path, seq_len=20, seq_mode=True, training=True, test_ratio=0.01,episode_length=None):
        self.training = training
        self.episode_length = episode_length
        self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
        np.random.seed(0)
        # print("ddddddddddddddddddddddddddddffffffffffffffffffffffffffffffffffff",self.episode_length)

        indices = np.arange(0, len(self.fpaths))

        n_trainset = int(len(indices)*(1.0-test_ratio))

        self.train_indices = indices[:n_trainset]
        # print("train_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indic ",self.train_indices)

        self.test_indices = indices[n_trainset:]
        # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
        # self.test_indices = np.delete(indices, self.train_indices)
        self.indices = self.train_indices if training else self.test_indices
        self.seq_len = seq_len
        self.seq_mode = seq_mode
        # print("seq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_mode ",seq_len)

    def __getitem__(self, idx):


        def pad_tensor(t, episode_length=self.episode_length, window_length=self.seq_len-1, pad_function=torch.zeros):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            pad_size = episode_length - t.size(0) + window_length
            # Add window lenght - 1 infront of the number of obersavtion
            # begin_pad       = pad_function([window_length-1, t.size(1)]).to(device)
            # pad the environment with lenght of the episode subtracted from  the total episode length
            episode_end_pad = pad_function([pad_size,      t.size(1)]).to(device)
            # print("paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",episode_end_pad.shape)

            return torch.cat([t.to(device),episode_end_pad], dim=0)



        npz = np.load(self.fpaths[self.indices[idx]])
        # print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr", idx )

        obs = npz['obs'] # (T, H, W, C) np array
        # print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111",obs.shape)

        obs = torch.from_numpy(obs) 
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",obs.shape)
        obs = pad_tensor(obs ,window_length=(self.seq_len-1)).cpu().detach().numpy()
        # obs = torch.from_numpy(obs) 


        actions = npz['action'] # (T, n_actions) np array
        # print("2222222222222222222222222222222222222222222222222222222222222222",actions.shape)

        actions = torch.from_numpy(actions) 

        actions = pad_tensor(actions, window_length=(self.seq_len-1)).cpu().detach().numpy()
        # actions=torch.from_numpy(actions)
        # print("8888888888888888888888888888888888888888888888888888888888888888888888888888888", obs )
        # print("9999999999999999999999999999999999999999999999999999999999999999999999999", obs.shape )

        T, C = obs.shape
        # T, H, W, C = obs.shape
        n_seq = T // self.seq_len
        end_seq = n_seq * self.seq_len # T' = end of sequence
        # print("0000000000000000000000000000000000000000000000000000000000000000000 ",n_seq )
        # print("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",self.seq_len )
        # print("cccccccccccccccccccccccccccccccccccccccccccccccccccc",obs.shape)
        # print("dssssssssssssssssssssssssssssssssssssssssssssssssssss",actions.shape)   
        #      
        # print("end_seqend_seqend_seqend_seqend_seqend_seqend_seq",end_seq)          
        obs = obs[:end_seq].reshape([-1, self.seq_len, C]) # (N_seq, seq_len, H, W, C)
        actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]]) # 

        # if args.seq_mode:
        #     start_range = max_len-self.seq_len
        #     for t in range(0, max_len-self.seq_len, self.seq_len):
        #         obs[t:t+self.seq_len]
        # else:
        #     rand_start = np.random.randint(max_len-self.seq_len)
        #     obs = obs[rand_start:rand_start+self.seq_len] # (T, H, W, C)
        #     actions = actions[rand_start:rand_start+self.seq_len]
        return obs, actions

    def __len__(self):
        return len(self.indices)
# class GameSceneDataset_reward(torch.utils.data.Dataset):
#     def __init__(self, data_path, seq_len=10, seq_mode=True, training=True, test_ratio=0.5):
#         self.training = training
#         self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
#         np.random.seed(0)

#         indices = np.arange(0, len(self.fpaths))
#         # print("tttttttttttttttttttttttttttttttindicesindicesindicesindicesindicesindicesindicesindicesttttttttttttttttttttttttttttttttttttttttttttttttttttttt ",indices)

#         n_trainset = int(len(indices)*(1.0-test_ratio))

#         self.train_indices = indices[:n_trainset]
#         # print("train_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indic ",self.train_indices)

#         self.test_indices = indices[n_trainset:]
#         # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
#         # self.test_indices = np.delete(indices, self.train_indices)
#         self.indices = self.train_indices if training else self.test_indices
#         self.seq_len = seq_len
#         self.seq_mode = seq_mode
#         # print("seq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_mode ",seq_len)

#     def __getitem__(self, idx):
#         npz = np.load(self.fpaths[self.indices[idx]])
#         # print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr", idx )

#         obs = npz['obs'] # (T, H, W, C) np array
#         actions = npz['action']
#         reward = npz['reward']  # (T, n_actions) np array
#         # print("8888888888888888888888888888888888888888888888888888888888888888888888888888888", obs )
#         # print("9999999999999999999999999999999999999999999999999999999999999999999999999", obs.shape )

#         # T, C = obs.shape
#         # # T, H, W, C = obs.shape
#         # n_seq = T // self.seq_len
#         # end_seq = n_seq * self.seq_len # T' = end of sequence
#         # # print("0000000000000000000000000000000000000000000000000000000000000000000 ",n_seq )
#         # # print("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",self.seq_len )
#         # # print("cccccccccccccccccccccccccccccccccccccccccccccccccccc",obs.shape)
#         # # print("dssssssssssssssssssssssssssssssssssssssssssssssssssss",actions.shape)        
#         # # print("end_seqend_seqend_seqend_seqend_seqend_seqend_seq",end_seq)          
#         # obs = obs[:end_seq].reshape([-1, self.seq_len, C]) # (N_seq, seq_len, H, W, C)
#         # actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]]) # 

#         # if args.seq_mode:
#         #     start_range = max_len-self.seq_len
#         #     for t in range(0, max_len-self.seq_len, self.seq_len):
#         #         obs[t:t+self.seq_len]
#         # else:
#         #     rand_start = np.random.randint(max_len-self.seq_len)
#         #     obs = obs[rand_start:rand_start+self.seq_len] # (T, H, W, C)
#         #     actions = actions[rand_start:rand_start+self.seq_len]
#         return obs, actions, reward

#     def __len__(self):
#         return len(self.indices)













##################################################################################

# class GameEpisodeDatasetDream(torch.utils.data.Dataset):
#     def __init__(self, data_path, seq_len=10, seq_mode=True, training=True, test_ratio=0.10):
#         self.training = training
#         self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
#         np.random.seed(0)

#         indices = np.arange(0, len(self.fpaths))
#         # print("tttttttttttttttttttttttttttttttindicesindicesindicesindicesindicesindicesindicesindicesttttttttttttttttttttttttttttttttttttttttttttttttttttttt ",indices)

#         n_trainset = int(len(indices)*(1.0-test_ratio))

#         self.train_indices = indices[:n_trainset]
#         # print("train_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indicestrain_indic ",self.train_indices)

#         self.test_indices = indices[n_trainset:]
#         # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
#         # self.test_indices = np.delete(indices, self.train_indices)
#         self.indices = self.train_indices if training else self.test_indices
#         self.seq_len = seq_len
#         self.seq_mode = seq_mode
#         # print("seq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_modeseq_mode ",seq_len)

    # def __getitem__(self, idx):
    #     npz = np.load(self.fpaths[self.indices[idx]])
    #     # print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr", idx )

    #     obs = npz['obs'] # (T, H, W, C) np array
    #     actions = npz['action'] 
    #     reward = npz['reward'] # (T, n_actions) np array
    #     # print("8888888888888888888888888888888888888888888888888888888888888888888888888888888", len(reward.shape ))

    #     T, C = obs.shape
    #     # T, H, W, C = obs.shape
    #     n_seq = T // self.seq_len
    #     end_seq = n_seq * self.seq_len # T' = end of sequence

    #     # print("cccccccccccccccccccccccccccccccccccccccccccccccccccc",actions.shape)

    #     # print("0000000000000000000000000000000000000000000000000000000000000000000 " )
    #     # print("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",actions.shape[-1] )
    #     # print("dssssssssssssssssssssssssssssssssssssssssssssssssssss",actions.shape)        
    #     # print("end_seqend_seqend_seqend_seqend_seqend_seqend_seq",end_seq)          
    #     obs = obs[:end_seq].reshape([-1, self.seq_len, C]) # (N_seq, seq_len, H, W, C)
    #     actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]]) # 
        
    #     reward = reward[:end_seq].reshape([-1, self.seq_len, 1]) # 

    #     # if args.seq_mode:
    #     #     start_range = max_len-self.seq_lensss
    #     #     for t in range(0, max_len-self.seq_len, self.seq_len):
    #     #         obs[t:t+self.seq_len]
    #     # else:
    #     #     rand_start = np.random.randint(max_len-self.seq_len)
    #     #     obs = obs[rand_start:rand_start+self.seq_len] # (T, H, W, C)
#     #     #     actions = actions[rand_start:rand_start+self.seq_len]
#     #     return obs, actions, reward

#     def __len__(self):
#         return len(self.indices)

# # def collate_dream(data):
#     # obs (B, N_seq, seq_len, H, W, C), actions (B, N_seq, seq_len, n_actions)
#     obs, actions ,reward= zip(*data)
#     obs, actions, reward = np.array(obs), np.array(actions) ,np.array(reward) 
#     # print("llllllllllllllllllllllllllllllllllllllllllllllllllllll",obs.shape)

#     _,_, seq_len, C = obs.shape

#     obs = obs.reshape([-1, C]) # (B*N_seq*seq_len, H, W, C)
#     actions = actions.reshape([-1, seq_len, actions.shape[-1]]) # (B*n_seq, n_actions)
#     reward = reward.reshape([-1, seq_len, 1]) # (B*n_seq, n_actions)
#     obs_lst = []
#     for i in range(len(obs)): # batch loop
#         # print(obs[i])
#         obs_lst.append(torch.from_numpy(obs[i]))
#         # obs_lst.append(transform(obs[i]))
#         # for j in range(len(obs[i])): # sequence loop
#         #     obs_lst.append(transform(obs[i][j]))
#     obs = torch.stack(obs_lst, dim=0) # (B*N_seq*seq_len, C, H, W)
#     # obs = obs.view([-1, seq_len, H, W, C]) # (B*N_seq, seq_len, C, H, W)
#     return obs, torch.tensor(actions, dtype=torch.float) ,torch.tensor(reward, dtype=torch.float)

