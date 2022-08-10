
import gym
import numpy as np
from ENVIRONMENT import socnavenv
from ENVIRONMENT.socnavenv import SocNavEnv
import random
import os
import argparse

parser = argparse.ArgumentParser("rollouts asigning")
parser.add_argument('--rollouts', type=int,
                    help="Number of rollouts.")
parser.add_argument('--testrollouts', type=int,
                    help="Number of rollouts.")

args = parser.parse_args()
rollout_dir = 'data'
if not os.path.exists('data'):
    os.makedirs('data')


def generate_test_data(rollouts): 

    rollouts = (rollouts)
    """ Generates data """

    env = SocNavEnv()
    prv_s_rollout = []
    s_rollout = []
    r_rollout = []
    d_rollout = []
    for i in range(rollouts):
        prev_s= env.reset()
        
        t = 0
        while True:
            action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) #a_rollout[t]
            t += 1
            prv_s_rollout += [prev_s]
            s, r, done, _ = env.step(action)
            #env.render()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            prev_s = s
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                break


    np.savez('./data/vae_test_data',
                observations=np.array(s_rollout),
                rewards=np.array(r_rollout),
                actions=np.array(action),
                terminals=np.array(d_rollout),
                prv_s_rollout = np.array(prv_s_rollout))

                
def generate_data(rollouts): 
    """ Generates data """

    env = SocNavEnv()
    prv_s_rollout = []
    s_rollout = []
    r_rollout = []
    d_rollout = []
    for i in range(rollouts):
        prev_s= env.reset()
        
        t = 0
        while True:
            action = np.array([random.uniform(-1, 1), random.uniform(-1, 1)]) #a_rollout[t]
            t += 1
            prv_s_rollout += [prev_s]
            s, r, done, _ = env.step(action)
            #env.render()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            prev_s = s
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                break


    np.savez('./data/vae_train_data',
                observations=np.array(s_rollout),
                rewards=np.array(r_rollout),
                actions=np.array(action),
                terminals=np.array(d_rollout),
                prv_s_rollout = np.array(prv_s_rollout))
    





generate_data(args.rollouts)
generate_test_data(args.testrollouts)