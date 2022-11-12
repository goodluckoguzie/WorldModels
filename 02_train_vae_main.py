import torch
import torch.nn as nn
import numpy as np
# from hparams import VAEHyperParams as hp
# from models import VAE, vae_loss
from torch.utils.data import DataLoader
from data import *
from tqdm import tqdm
import os, sys
from torchvision.utils import save_image
from torch.nn import functional as F
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml
from UTILITY.early_stopping_for_vae import  EarlyStopping

DEVICE = None


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





class VAE_MODEL():
    def __init__(self, config:str, **kwargs) -> None:
        assert(config is not None)
        # initializing the env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extra = None
        self.data_dir = None
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
        # print("dddddddddddddddddddddd",self.run_name )

                # setting values from config file
        self.configure(self.config)


        # self.RNN  = LSTM(self.n_latents, self.n_actions, self.n_hiddens).to(device)

        # declaring the network
        global_step = 0
        self.model = VAE(self.n_latents,self.n_hiddens,self.n_latents).to(DEVICE)

  
        self.data_path = self.data_dir# if not self.extra else self.extra_dir

        self.dataset = GameSceneDataset(self.data_path)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.n_workers,)
        print("Train dataset lenght ",len(self.loader))

        self.validset = GameSceneDataset(self.data_path, training=False)
        self.valid_loader = DataLoader(self.validset, batch_size=self.test_batch, shuffle=False, drop_last=True)
        print("valid dataset lenght ",len(self.valid_loader))

        self.ckpt_dir = os.path.join(self.ckpt_dir, 'vae')
        self.sample_dir = os.path.join(self.ckpt_dir, 'samples')
        os.makedirs(self.sample_dir, exist_ok=True)



        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=6e-4)
        
    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.extra is None:
            self.extra = config["extra"]
            assert(self.extra is not None), f"Argument extra size cannot be None"

        if self.data_dir is None:
            self.data_dir = config["data_dir"]
            assert(self.data_dir is not None), f"Argument data_dir  cannot be None"

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
        VAE_runs = 'VAE_runs'
        if not os.path.exists(VAE_runs):
            os.makedirs(VAE_runs)
        if self.run_name is not None:
            self.writer = SummaryWriter('VAE_runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

        self.early_stopping = EarlyStopping(patience=100, verbose=True)



    def plot(self, episode):
        self.Train_loss.append(self.train_loss)
        self.Valid_loss.append(self.valid_loss)
        self.grad_norms.append(self.total_grad_norm/self.batch_size)


        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.total_grad_norm/self.batch_size), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "Train_loss"), np.array(self.train_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "Valid_loss"), np.array(self.valid_loss), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("Train_loss / epoch", self.train_loss, episode)
        self.writer.add_scalar("valid_loss / epoch", self.valid_loss, episode)
        self.writer.add_scalar("Average total grad norm / episode", (self.total_grad_norm/self.batch_size), episode)





    def train(self):


        def evaluate(self):
            self.model.eval()
            valid_losses = []
            total_recon_loss = []
            total_kld_loss = []

            with torch.no_grad():
                for idx, obs in enumerate(self.valid_loader):
                    x = obs.to(DEVICE)
                    # import pdb; pdb.set_trace()
                    x_hat, mu, logvar = self.model(x)
                    valid_loss, recon_loss, kld = vae_loss(x_hat, x, mu, logvar)

                    total_recon_loss.append(recon_loss.item())
                    total_kld_loss.append(kld.item())
                    valid_losses.append(valid_loss.item())
                    
            total_kld_loss =np.mean(total_kld_loss)
            self.model.train()
            return valid_losses, total_recon_loss, total_kld_loss

        global_step = 0
        # model = self.VAE(47,256,47).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        # Loaded pretrained VAE
        # ckpts = sorted(glob.glob(os.path.join(hp.ckpt_dir, 'vae', '*k.pth.tar')))
        # if ckpts:
        #     ckpt = ckpts[-1]
        #     vae_state = torch.load(ckpt)
        #     model.load_state_dict(vae_state['model'])
        #     global_step = int(os.path.basename(ckpt).split('.')[0][:-1]) * 1000
        #     print('Loaded vae ckpt {}'.format(ckpt))

        data_path = self.data_dir 
        dataset = GameSceneDataset(data_path)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_workers,
        )
        # print("hp.batch_sizehp.batch_sizehp.batch_sizehp.batch_size",self.batch_size)
        # print("hp.vsizehp.vsizehp.vsizehp.vsizehp.vsizehp.vsizehp.vsize",self.vsize)
        self.ckpt_dir = os.path.join(self.ckpt_dir, 'vae')
        sample_dir = os.path.join(self.ckpt_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        # print(len(loader))
        total_grad_norm = 0
        
        while global_step <  self.max_step:

            self.Train_loss = []
            self.Valid_loss = []
            self.grad_norms = []
            self.global_step = 0
            self.train_losses = []
            self.valid_losses = []
            self.total_grad_norm = 0   

            for idx, obs in enumerate(tqdm(loader, total=len(loader))):
                x = obs.to(DEVICE)
                x_hat, mu, logvar = self.model(x)
                
                loss, recon_loss, kld = vae_loss(x_hat, x, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                total_grad_norm += (torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5).cpu())#/w
                # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh,",total_grad_norm )
                optimizer.step()
                self.train_losses.append(loss.item())

                self.valid_losses,self.total_recon_loss, self.total_kld_loss = evaluate(self)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # with open(os.path.join(self.ckpt_dir, 'train.log'), 'a') as f:
                #     log = '{} || Step: {}, loss: {:.4f}, kld: {:.4f}\n'.format(now, self.global_step, recon_loss, self.total_kld_loss)
                #     f.write(log)
                epoch_len = len(str(self.global_step))

                self.train_loss = np.mean(self.train_losses)/len(self.loader)
                self.valid_loss = np.mean(self.valid_losses)/len(self.valid_loader)
                # self.kld = (self.kld )/len(self.valid_loader)

                print_msg = (f'[{self.global_step:>{epoch_len}}/{self.global_step:>{epoch_len}}] ' +
                            f'train_loss: {self.train_loss:.8f} ' +
                            f'valid_loss: {self.valid_loss:.8f}')
                            
                self.plot(self.global_step +1)


                if self.global_step % self.save_interval == 0:
                    d = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    torch.save(
                        d, os.path.join(self.ckpt_dir, '{:03d}k.pth.tar'.format(self.global_step//1000))
                        )

                    
                self.global_step += 1
                        # clear lists to track next epoch
                self.train_losses = []
                self.valid_losses = []

                    # and if it has, it will make a checkpoint of the current model
                if self.global_step % 10 == 0:
                    self.early_stopping(self.valid_loss, self.model)
                    print(print_msg)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
                



    

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config file for the model
    config = "./configs/VAE_model.yaml"
        # declaring the network
    Agent = VAE_MODEL(config, run_name="VAE")


    # print(config)
    Agent.train()

