
from genericpath import sameopenfile
import torch
from torch import nn
from torch.autograd import Variable
import numpy
from os import listdir
import os
import sys
from matplotlib.pyplot import axis
import random
import gym
from gym import spaces
import cv2
import numpy as np
import math

from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from tqdm import tqdm



NORMALISE_FACTOR_POS = 40.*2
NORMALISE_FACTOR_SPEED = 5.*2
NORMALISE_COS_SIN = 2.

def get_observation_from_dataset(dataset, idx):
    sample = dataset[idx,:]
    return transform_processed_observation_into_raw(sample)

def transform_processed_observation_into_raw(sample):
    ind_pos = [0,1,2]
    robot_obs = [ sample[i] for i in ind_pos ]
    robot_obs[2] = math.atan2(sample[3], sample[2])
    robot_obs = np.array(robot_obs)
    ind_pos = [4,5]
    goal_obs = [sample[i] for i in ind_pos]
    goal_obs = np.array(goal_obs)
    humans = []
    for human_num in range(5):
        offset = 6 + human_num*5
        human_angle = math.atan2(sample[offset+3], sample[offset+2])
        humans.append([sample[offset+0], sample[offset+1], human_angle, sample[offset+4]])
    humans_obs = np.array(humans)

    return robot_obs, goal_obs, humans_obs


def test_data(batch_size):
    with numpy.load('./data/Test_dataset.npz') as data:
        a = data['observations.npy']
    x = numpy.array(a)
    #x =x[:1,:]
    #test_data = []
    #for i in range(len(x)):
        #test_data.append(x[i]) 
    #test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

    return x




def normalised(sample, factor_pos=1./NORMALISE_FACTOR_POS, factor_vel=1./NORMALISE_FACTOR_SPEED, factor_cossin=1./NORMALISE_COS_SIN, constant_0=0., constant_1=0.5):
    ret = np.array(sample)

    ret += constant_0

    # Make the data go from -0.5 to 0.5
    ret[:, 0] = ret[:, 0] * factor_pos
    ret[:, 1] = ret[:, 1] * factor_pos
    ret[:, 2] = ret[:, 2] * factor_cossin
    ret[:, 3] = ret[:, 3] * factor_cossin
    ret[:, 4] = ret[:, 4] * factor_pos
    ret[:, 5] = ret[:, 5] * factor_pos

    for human_num in range(5):
        offset = 6 + human_num*5
        ret[:, offset+0] = ret[:, offset+0] * factor_pos
        ret[:, offset+1] = ret[:, offset+1] * factor_pos
        ret[:, offset+2] = ret[:, offset+2] * factor_cossin
        ret[:, offset+3] = ret[:, offset+3] * factor_cossin
        ret[:, offset+4] = ret[:, offset+4] * factor_vel

    # Add constant (0.5 to normalise) to make the data go from 0 to 1
    ret += constant_1

    return ret

def denormalised(sample):
    return normalised(sample=sample, factor_pos=NORMALISE_FACTOR_POS, factor_vel=NORMALISE_FACTOR_SPEED, factor_cossin=NORMALISE_COS_SIN, constant_0=-0.5, constant_1=-0.)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 31
z_dim = 25
"""
def transform_output_into_observation1():

    # load your model architecture/module
    model = VAE().to(device)
    # fill your architecture with the trained weights
    model.load_state_dict(torch.load('./saved_model/TVAE_model.pt'))
    data = get_input_data() 
    data = Variable(torch.from_numpy(data), requires_grad=False)
    model.eval()

    #data1 = data.view(data.size(0), -1)
    #zs = model.get_z(data).data.numpy()
"""


class Logger:
    def __init__(self, path):
        self.f = open(path, 'w')

    def __del__(self):
        self.f.close()

    def write(self, text):
        self.f.write(text+"\n")
        self.f.flush()
        print(text)
        
def plot_loss(dirname, history):
    """ Plot loss """

    plot_train = np.array(history["train"])
    is_val = ("test" in history.keys())
    if is_val:
        plot_val = np.array(history["test"])

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # epochs
    n_epochs = len(plot_train)

    # X axis
    x = [i for i in range(1, n_epochs+1)]
    
    # plot loss
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    plt.xlabel('epoch')
    plt.plot(x, plot_train[:], label='train loss')
    if is_val:
        plt.plot(x, plot_val[:], label='test loss')

    plt.legend()
    plt.savefig(os.path.join(dirname,'loss.png'))
    plt.close()



def code_and_decode(model, data):
    data = torch.from_numpy(data)
    data = Variable(data, requires_grad=False).to(device)
    model.eval()
    with torch.no_grad():
        zs= model.get_z(data)#.data.numpy()
        output = model.decode(zs ).cpu()

    return output 
