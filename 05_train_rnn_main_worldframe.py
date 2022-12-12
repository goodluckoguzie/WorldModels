import torch
import torch.nn as nn
import numpy as np
from hparams import RNNHyperParams as hp 
from hparams import NonPrePaddedWorldFrame_Datasets_Timestep_0_5 as data
from hparams import Seq_Len as Seq_len

# from models import VAE, RNN
from torch.utils.data import DataLoader
from data import *
from tqdm import tqdm
import os, sys
# from torchvision.utils import save_image
from torch.nn import functional as F
from datetime import datetime

DEVICE = None

from torch.utils.tensorboard import SummaryWriter
import yaml
from UTILITY.early_stopping_for_rnn import  EarlyStopping



class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, input_dims)

        self.input_dims = input_dims
    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = self.linear3(z)
        # z = torch.sigmoid(z)
        return z.reshape((-1, self.input_dims))


class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dims,  hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, hidden_dims)
        self.linear4 = nn.Linear(hidden_dims, latent_dims)
        self.linear5 = nn.Linear(hidden_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu =  self.linear4(x)
        sigma = torch.exp(self.linear5(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z ,mu , sigma





class VAE(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_dims, hidden_dims, latent_dims)
        self.decoder = Decoder(input_dims, hidden_dims, latent_dims)

    def forward(self, x):
        z,mu , sigma = self.encoder(x)
        return self.decoder(z),mu , sigma

    def vae_loss(recon_x, x, mu, logvar):
        """ VAE loss function """
        recon_loss = nn.MSELoss(size_average=False)
        BCE = recon_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD, BCE, KLD



class RNN(nn.Module):
    def __init__(self, n_latents, n_actions, n_hiddens):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_latents+n_actions, n_hiddens, batch_first=True)
        # target --> next latent (vision)
        self.fc = nn.Linear(n_hiddens, n_latents)

    def forward(self, states):
        h, _ = self.rnn(states)
        y = self.fc(h)
        return y, None, None
    
    def infer(self, states, hidden):
        h, next_hidden = self.rnn(states, hidden) # return (out, hx, cx)
        y = self.fc(h)
        return y, None, None, next_hidden





class RNN_MODEL():
    def __init__(self, config:str, **kwargs) -> None:
        assert(config is not None)
        # initializing the env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extra = None
        # self.data_dir = None
        self.extra_dir = None
        self.ckpt_dir = None
        # rnn variables
        self.n_latents = None
        self.n_actions = None
        self.n_hiddens = None
        self.batch_size = None
        self.test_batch = None
        self.n_sample = None
        self.log_interval = None
        self.save_interval = None
        self.max_step = None
        self.save_path = None
        self.n_workers = None
        self.run_name = None
        # self.seq_len = None
        self.run_name = None
        self.window = Seq_len.seq_8

                # setting values from config file
        self.configure(self.config)
        # declaring the network
        global_step = 0
        self.vae = VAE(self.n_latents,self.n_hiddens,self.n_latents).to(DEVICE)


        self.rnn = RNN(self.n_latents, self.n_actions, self.n_hiddens).to(DEVICE)
  
        # self.data_path = self.data_dir# if not self.extra else self.extra_dir
        self.ckpt_dir = data.ckpt_dir#'ckpt'
        self.rnnsave = data.rnnsave#'ckpt'
        self.data_path = data.data_dir 
        self.seq_len = Seq_len.seq_len_8
        episode_length = data.time_steps

        print(self.seq_len) 
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.rnnsave + self.window)

        # self.ckpt = sorted(glob.glob(os.path.join(self.ckpt_dir, 'vae', '*k.pth.tar')))[-1]

        # self.vae_state = torch.load(self.ckpt)
        # self.vae.load_state_dict(self.vae_state['model'])
        # self.vae.eval()
        # print('Loaded vae ckpt {}'.format(self.ckpt))       
   
        # self.ckpt  = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'rnn', '*.pth.tar')))[-1]
        # rnn_state = torch.load( self.ckpt, map_location={'cuda:0': str(self.device)})
        # print('Loaded rnn_state ckpt {}'.format(self.ckpt))
        # self.data_path = hp.data_dir 

        # dataset = GameEpisodeDataset(self.data_path, seq_len=self.seq_len,episode_length=episode_length)
        dataset = GameEpisodeDatasetNonPrePadded(self.data_path, seq_len=self.seq_len,episode_length=episode_length)

        self.loader = DataLoader(
            dataset, batch_size=1, shuffle=True, drop_last=True,
            num_workers=self.n_workers, collate_fn=collate_fn
        )
        # testset = GameEpisodeDataset(self.data_path, seq_len=self.seq_len, training=False,episode_length=episode_length)
        #Non pre-padde observation
        testset = GameEpisodeDatasetNonPrePadded(self.data_path, seq_len=self.seq_len, training=False,episode_length=episode_length)

        self.valid_loader = DataLoader(
            testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn
        )

        # self.ckpt_dir = os.path.join(self.ckpt_dir, 'rnn')
        sample_dir = os.path.join(self.ckpt_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=6e-4)
        
    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.extra is None:
            self.extra = config["extra"]
            assert(self.extra is not None), f"Argument seq_len size cannot be None"


        # if self.seq_len is None:
        #     self.seq_len = config["seq_len"]
        #     assert(self.seq_len is not None), f"Argument extra size cannot be None"

        # if self.data_dir is None:
        #     self.data_dir = config["data_dir"]
        #     assert(self.data_dir is not None), f"Argument data_dir  cannot be None"

        if self.extra_dir is None:
            self.extra_dir = config["extra_dir"]
            assert(self.extra_dir is not None), f"Argument extra_dir cannot be None"

        if self.ckpt_dir is None:
            self.ckpt_dir = config["ckpt_dir"]
            assert(self.ckpt_dir is not None), f"Argument ckpt_dir  cannot be None"

        if self.n_latents is None:
            self.n_latents = config["n_latents"]
            assert(self.n_latents is not None), f"Argument n_latents size cannot be None"

        if self.n_hiddens is None:
            self.n_hiddens = config["n_hiddens"]
            assert(self.n_hiddens is not None), f"Argument hidden_layers cannot be None"

        if self.n_actions is None:
            self.n_actions = config["n_actions"]
            assert(self.n_actions is not None), f"Argument n_actions cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), f"Argument batch_size cannot be None"

        if self.test_batch is None:
            self.test_batch = config["test_batch"]
            assert(self.test_batch is not None), f"Argument test_batch cannot be None"

        if self.n_sample is None:
            self.n_sample = config["n_sample"]
            assert(self.n_sample is not None), f"Argument n_sample cannot be None"

        if self.log_interval is None:
            self.log_interval = config["log_interval"]
            assert(self.log_interval is not None), f"Argument log_interval cannot be None"


        if self.save_interval is None:
            self.save_interval = config["save_interval"]
            assert(self.save_interval is not None), f"Argument save_interval cannot be None"


        if self.max_step is None:
            self.max_step = config["max_step"]
            assert(self.max_step is not None), f"Argument max_step cannot be None"

        if self.save_path is None:
            self.save_path = config["save_path"]
            assert(self.save_path is not None), f"Argument save_path cannot be None"

        if self.n_workers is None:
            self.n_workers = config["n_workers"]
            assert(self.n_workers is not None), f"Argument n_workers cannot be None"

        if self.run_name is None:
            self.run_name = config["run_name"]
            assert(self.run_name is not None), f"Argument run_name cannot be None"


        # check vae dir exists, if not, create it
        RNN_runs = data.RNN_runs#'WorldFrame_RNN_model_runs'

        if not os.path.exists(RNN_runs):
            os.makedirs(RNN_runs)
        if self.run_name is not None:
            # self.writer = SummaryWriter('WorldFrame_RNN_model_runs/'+self.run_name)
            self.writer = SummaryWriter('RNN_model_runs/'+RNN_runs + self.window)
        else:
            self.writer = SummaryWriter()

        self.early_stopping = EarlyStopping(patience=100, verbose=True)
        self.best_score = 0


    def plot(self, episode):
        self.Train_loss.append(self.train_loss)
        self.Valid_loss.append(self.valid_loss)
        self.grad_norms.append(self.total_grad_norm/self.batch_size)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))


        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.total_grad_norm/self.batch_size), allow_pickle=True, fix_imports=True)

        np.save(os.path.join(self.save_path, "plots", "Train_loss"), np.array(self.train_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "Valid_loss"), np.array(self.valid_loss), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("Train_loss / epoch", self.train_loss, episode)
        self.writer.add_scalar("valid_loss / epoch", self.valid_loss, episode)
        self.writer.add_scalar("Average total grad norm / episode", (self.total_grad_norm/self.batch_size), episode)




    def train(self):
        self.Train_loss = []
        self.Valid_loss = []
        self.grad_norms = []
        # to track the validation loss as the model trains
        self.global_step = 0
        self.train_losses = []
        self.valid_losses = []
        # vae = VAE(hp.vsize,hp.n_hiddens,hp.vsize).to(DEVICE)

        self.l1 = nn.L1Loss()
        def evaluate(self):
            self.rnn.eval()
            self.total_loss = []
            l1 = nn.L1Loss()
            with torch.no_grad():
                for idx, (obs, actions) in enumerate(self.valid_loader):
                    # obs = torch.from_numpy(obs)
                    obs, actions = obs.to(DEVICE), actions.to(DEVICE)
                    # z,latent_mu, latent_var = self.vae(obs) # (B*T, vsize)
                    z = obs
                    
                    # z = vae.reparam(latent_mu, latent_var) # (B*T, vsize)
                    z = z.view(-1, self.seq_len, self.n_latents) # (B*n_seq, T, vsize)

                    next_z = z[:, 1:, :]
                    z, actions = z[:, :-1, :], actions[:, :-1, :]
                    states = torch.cat([z, actions], dim=-1) # (B, T, vsize+asize)
                    # states = torch.cat([GO_states, next_states[:,:-1,:]], dim=1)
                    x, _, _ = self.rnn(states)
                    
                    loss = self.l1(x, next_z)
        
                    self.total_loss.append(loss.item())
            self.rnn.train()
            return np.mean(self.total_loss)

        for idx in range(1, self.max_step + 1):        # while self.global_step < self.max_step:
            self.Train_loss = []
            self.Valid_loss = []
            self.grad_norms = []
            self.train_losses = []
            self.valid_losses = []
            self.total_grad_norm = 0  

            for idx, (obs, actions) in enumerate(self.loader):
                # for idx, (obs, actions) in t:


                with torch.no_grad():
                    # obs = torch.from_numpy(obs)
                    obs, actions = obs.to(DEVICE), actions.to(DEVICE)

                    # z,latent_mu, latent_var = self.vae(obs) # (B*T, vsize)
                    z = obs

                    # z = latent_mu
                    z = z.view(-1, self.seq_len, self.n_latents) # (B*n_seq, T, vsize)

                next_z = z[:, 1:, :]
                z, actions = z[:, :-1, :], actions[:, :-1, :]      
            
                states = torch.cat([z, actions], dim=-1) # (B, T, vsize+asize)
                x, _, _ = self.rnn(states)

                loss = self.l1(x, next_z)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.train_losses.append(loss.item())
            self.global_step += 1

            self.total_grad_norm += (torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), max_norm=0.5).cpu())/self.global_step

            if self.global_step % 1 == 0:
                self.valid_losses = evaluate(self)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(os.path.join(self.ckpt_dir, 'train.log'), 'a') as f:
                    log = '{} || Step: {}, train_loss: {:.4f}, valid_losses: {:.4f}\n'.format(now, self.global_step, loss.item(), self.valid_losses)
                    f.write(log)

            self.epoch_len = len(str(self.global_step))
            self.train_loss = np.mean(self.train_losses)/len(self.loader)
            self.valid_loss = self.valid_losses/len(self.valid_loader)
            self.plot(self.global_step +1)

            print_msg = (f'[{self.global_step:>{self.epoch_len}}/{self.global_step:>{self.epoch_len}}] ' +
                        f'train_loss: {self.train_loss:.8f} ' +
                        f'valid_loss: {self.valid_loss:.8f}')
            

                # clear lists to track next epoch
            self.train_losses = []
            self.valid_losses = []

            if self.global_step % self.save_interval == 0:
                d = {
                    'model': self.rnn.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(
                    d, os.path.join(self.ckpt_dir, '{:03d}worldframe.pth.tar'.format(self.global_step//self.save_interval))
                )

                # and if it has, it will make a checkpoint of the current model
            if self.global_step % self.save_interval == 0:
                self.early_stopping(self.valid_loss, self.rnn)

                # if self.global_step == 0:#is None:
                #     self.best_score = self.valid_loss
                #     d = {
                #         'model': self.rnn.state_dict(),
                #         'optimizer': self.optimizer.state_dict(),
                #     }
                #     torch.save(
                #         d, os.path.join(self.ckpt_dir, '{:03d}robotframe.pth.tar'.format(self.global_step//self.save_interval))
                #     )                
                # elif self.valid_loss < self.best_score :
                #     d = {
                #         'model': self.rnn.state_dict(),
                #         'optimizer': self.optimizer.state_dict(),
                #     }
                #     torch.save(
                #         d, os.path.join(self.ckpt_dir, '{:03d}robotframe.pth.tar'.format(self.global_step//self.save_interval))
                #     )      
                #     self.best_score = self.valid_loss 

                # else:
                #     pass

            if self.global_step % 50 == 0:
                print(print_msg)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
                

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(0)

    # config file for the model
    config = "./configs/worldframe_RNN_model.yaml"
        # declaring the network
    Agent =RNN_MODEL(config, run_name="worldframe_RNN_model")


    # print(config)
    Agent.train()