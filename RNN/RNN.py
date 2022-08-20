import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.autograd import Variable

# class RNN(nn.Module):
#     def __init__(self, n_latents, n_actions, n_hiddens):
#         super(RNN, self).__init__()
#         self.n_latents = n_latents
#         self.n_actions = n_actions
#         self.n_hiddens = n_hiddens
#         self.rnn = nn.LSTM(n_latents+n_actions, n_hiddens, batch_first=True)
#         self.fc = nn.Linear(n_hiddens, n_latents)

#     def forward(self, states):
        
#         #h, _ = self.rnn(states)
#         h, (h_out, _)  = self.rnn(states)
#         h_out = h_out.view(-1, self.n_hiddens)
#         y = self.fc(h)
#         return y, h_out, None
    
#     def infer(self, states, hidden):
#         h, next_hidden = self.rnn(states, hidden) # return (out, hx, cx)
#         y = self.fc(h)
#         return y, None, None, next_hidden



class LSTM(nn.Module):
    num_layers = 3

    def __init__(self, n_latents, n_actions, n_hiddens,num_layers):
        super(LSTM, self).__init__()
        
        self.n_latents = n_latents
        self.n_actions = n_actions
        self.n_hiddens = n_hiddens
        self.num_layers = num_layers
        self.rnn = nn.LSTM(n_latents+n_actions, hidden_size=n_hiddens,
                            batch_first=True)
        self.fc = nn.Linear(n_hiddens, n_latents)

    def forward(self, states):
        h_0 = Variable(torch.zeros(
            self.num_layers, states.size(0), self.n_hiddens))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, states.size(0), self.n_hiddens))
        
        # Propagate input through LSTM
        h, (h_out, _) = self.rnn(states, (h_0, c_0))
        
        h_out = h_out.view(-1, self.n_hiddens)
        
        out = self.fc(h_out)
        y = self.fc(h)
        
        return y , out



# class RNN_REWARD_INCLUDED(nn.Module):
#     def __init__(self, n_latents, n_actions, reward,n_hiddens):
#         super(Rnn, self).__init__()
#         self.rnn = nn.LSTM(n_latents+n_actions+reward, n_hiddens, batch_first=True)
#         # target --> next observation (vision)
#         self.fc = nn.Linear(n_hiddens, n_latents)

#     def forward(self, states):
#         h, _ = self.rnn(states)
#         y = self.fc(h)
#         return y, None, None
    
#     def infer(self, states, hidden):
#         h, next_hidden = self.rnn(states, hidden) # return (out, hx, cx)
#         y = self.fc(h)
#         return y, None, None, next_hidden


