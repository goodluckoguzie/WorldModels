import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os, sys, glob

import random

class NeuralNetwork(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_shape, 32)
        self.l2 = nn.Linear(32, 32)
        self.lout = nn.Linear(32, n_actions)
        
    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        return self.lout(x)
    
    def get_params(self):
        p = np.empty((0,))
        for n in self.parameters():
            p = np.append(p, n.flatten().cpu().detach().numpy())
        return p
    
    def set_params(self, x):
        start = 0
        for p in self.parameters():
            e = start + np.prod(p.shape)
            p.data = torch.FloatTensor(x[start:e]).reshape(p.shape)
            start = e




def evaluate(ann, env, seed):
    env.seed(seed) # deterministic for demonstration
    obs = env.reset()
    total_reward = 0
    while True:
        env.render()
        # Output of the neural net
        net_output = ann(torch.tensor(obs))
        # the action is the value clipped returned by the nn
        action = net_output.data.cpu().numpy().argmax()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break
        # print("total_reward",total_reward)

    return total_reward



import cma
ENV_NAME = 'LunarLander-v2'
np.random.seed(123)
env = gym.make(ENV_NAME)
import datetime

now = datetime.datetime.now()

date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

writer_name = 'cat_{}_{}'.format(ENV_NAME, date_time)

writer = SummaryWriter(log_dir='runs/'+writer_name)

ann = NeuralNetwork(env.observation_space.shape[0], env.action_space.n)


es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1,{'popsize': 50,'seed': 123})

# es = cma.CMAEvolutionStrategy(len(ann.get_params()) * [0], 0.1, {'seed': 123})


def fitness(x, ann, env, seed, visul=False):
    ann.set_params(x)
    return -evaluate(ann, env, seed)




# best = 0
# for i in range(100):
#     solutions = np.array(es.ask())
#     fits = [fitness(x, ann, env) for x in solutions]

#     es.tell(solutions, fits)
#     es.disp()
#     cur_best = max(fits)
#     best_index = np.argmax(fits)
#     best_params = solutions[best_index]
#     print("current  value {}...".format(cur_best))

#     # print('current best reward : {}'.format(cur_best))
#     if not best or cur_best >= best:
#         best = cur_best
#         print("Saving new best with value {}...".format(cur_best))
#         d = best_params



# print('best reward : {}'.format(best))



best = 0
for iteration in range(10000):
    seed = random.getrandbits(32)
    # Create population of candidates and evaluate them
    candidates, fitnesses , Maxfitnesses = es.ask(), [],[]
    for candidate in candidates:
        # Load new policy parameters to agent.
        # ann.set_params(candidate)
        # Evaluate the agent using stable-baselines predict function
        reward = fitness(candidate, ann, env, seed) 
        fitnesses.append(reward)
        Maxfitnesses.append(-reward)
    # CMA-ES update
    es.tell(candidates, fitnesses)
    # Display some training infos
    mean_fitness = np.mean(sorted(fitnesses)[:int(0.1 * len(candidates))])
    print("Iteration {:<3} Mean top 10% reward: {:.2f}".format(iteration, -mean_fitness))
    
    cur_best = max(Maxfitnesses)
    best_index = np.argmax(Maxfitnesses)
    print("current  value {}...".format(cur_best))

    writer.add_scalar('Mean top 10 reward', -mean_fitness, iteration)
    writer.add_scalar('reward', cur_best, iteration)


    best_params = candidates[best_index]
    rew = evaluate(ann, env, random.getrandbits(32))
    writer.add_scalar('test reward', rew, iteration)
    # print('current best reward : {}'.format(cur_best))
    if not best or cur_best >= best:
        best = cur_best
        print("Saving new best with value {}...".format(cur_best))
        d = best_params
        torch.save(ann.state_dict(), 'cat.pt')
        # if i % 50 == 0:
        #     evaluate(ann, env,view=True)

    def save_model(ann ,path):
        torch.save(ann.state_dict(), path)
    save_path = "./models/cat"
    # saving model
    if (save_path is not None) and ((iteration+1)%5 == 0):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        try:
            save_model(ann ,os.path.join(save_path, "episode"+ str(iteration+1).zfill(8) + ".pth"))
        except:
            print("Error in saving model")

print('best reward : {}'.format(best))
