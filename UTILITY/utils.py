import torch
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


latents = 31
actions = 2
hiddens = 256
gaussians = 5


###################################################################################################################

def get_mixture_coef(z_pred):
    # print("gggggggggggggggggggggggggggggggggggg")
    # print(z_pred)
    # print(z_pred.shape)
    log_pi, mu, log_sigma = torch.split(z_pred, 2, 1)
    log_pi = log_pi - torch.log(torch.sum(torch.exp(log_pi), axis = 1, keepdims = True))

    return log_pi, mu, log_sigma

def get_pi_idx(x, pdf):
    # samples from a categorial distribution
    N = pdf.size()
    accumulate = 0
    for i in range(0, N[0]):
        accumulate += pdf[i]
        if (accumulate >= x):
            return i
    random_value = np.random.randint(N)
    #print('error with sampling ensemble, returning random', random_value)
    return random_value

def sample_z(mu, log_sigma):
    z =  mu + (torch.exp(log_sigma)) * torch.randn(*log_sigma.shape) 
    return z


def get_z_from_rnn_output(y_pred):
    HIDDEN_UNITS = 256
    GAUSSIAN_MIXTURES = 5
    Z_DIM = 31
    d = GAUSSIAN_MIXTURES * Z_DIM

    z_pred = y_pred[:,:(d)]
    rew_pred = y_pred[:,-1]

    z_pred = torch.reshape(z_pred, [-1, GAUSSIAN_MIXTURES])

    log_pi, mu, log_sigma = get_mixture_coef(z_pred)

    chosen_log_pi = torch.zeros(Z_DIM)
    chosen_mu = torch.zeros(Z_DIM)
    chosen_log_sigma = torch.zeros(Z_DIM)

    # adjust temperatures
    logmix2 = log_pi
    logmix2 -= logmix2.max()
    logmix2 = torch.exp(logmix2)
    logmix2 /= logmix2.sum(axis=1).reshape(Z_DIM, 1)


    for j in range(Z_DIM):
        idx = get_pi_idx(np.random.rand(), logmix2[j])
        chosen_log_pi[j] = idx
        chosen_mu[j] = mu[j, idx]
        chosen_log_sigma[j] = log_sigma[j,idx]

    next_z = sample_z(chosen_mu, chosen_log_sigma)
    
    # custom reward output for CMA-es
    next_reward = F.sigmoid(rew_pred)
    """
    if rew_pred > 0:
        next_reward = 1
    else:
        next_reward = 0
    """    
    return next_z, next_reward, chosen_mu













class _MDRNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, gaussians * latents + 1)

    def forward(self, *inputs):
        pass


class MDN_RNN(_MDRNNBase):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.rnn = nn.LSTM(1 + actions + latents, hiddens)
        self.gmm_linear = nn.Linear(hiddens, gaussians * latents + 1)
    def forward(self, ins):
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)
        
        return gmm_outs
    
class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(1 + latents + actions, hiddens)

    def forward(self, input, hidden): # pylint: disable=arguments-differ
        next_hidden = self.rnn(input, hidden)
        out_rnn = next_hidden[0]
        out_full = self.gmm_linear(out_rnn)

        return out_full, next_hidden



Z_FACTOR = 1
REWARD_FACTOR = 5

def get_responses(y_true):

    z_true = y_true[:,:,:latents]
    rew_true = y_true[:,:,-1]

    return z_true, rew_true




def lognormal(z_true, mu, log_sigma):

    logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
    return -0.5 * ((z_true - mu) / log_sigma.exp()) ** 2 - log_sigma - logSqrtTwoPI



def rnn_z_loss(y_true, y_pred):
    z_true, rew_true = get_responses(y_true) 
    d = gaussians * latents
    z_pred = y_pred[:,:,:(d)]
    z_pred = torch.reshape(z_pred, [-1, gaussians ])

    log_pi, mu, log_sigma = get_mixture_coef(z_pred)

    flat_z_true = torch.reshape(z_true,[-1, 1])

    z_loss = log_pi + lognormal(flat_z_true, mu, log_sigma)
    z_loss = -torch.log(torch.sum(z_loss.exp(), 1, keepdims=True))

    z_loss = torch.mean(z_loss) 

    return z_loss

def rnn_rew_loss(y_true, y_pred):
    z_true, rew_true = get_responses(y_true) #, done_true
    reward_pred = y_pred[:,:,-1]
    rew_loss = F.binary_cross_entropy(F.sigmoid(reward_pred), rew_true, reduce=False)
    rew_loss = torch.mean(rew_loss)

    return rew_loss

def rnn_loss(y_true, y_pred):

    z_loss = rnn_z_loss(y_true, y_pred)
    rew_loss = rnn_rew_loss(y_true, y_pred)

    return (Z_FACTOR * z_loss + REWARD_FACTOR * rew_loss) / (Z_FACTOR + REWARD_FACTOR)