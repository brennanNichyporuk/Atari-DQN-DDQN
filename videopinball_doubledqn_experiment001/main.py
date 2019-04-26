from comet_ml import Experiment, ExistingExperiment

import gym
from gym.envs import registry
from gym.envs.registration import register
from gym import logger as gymlogger
gymlogger.set_level(40) #error only
import torch.nn.functional as F
import numpy as np
import argparse
import shutil
import random
import warnings
import time
import matplotlib
import matplotlib.pyplot as plt
import skimage.transform as T
import math
import glob
import io
import base64
import os
from os.path import split, join, dirname, realpath

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

parser = argparse.ArgumentParser()
parser.add_argument('--api-key', type=str, default='', 
                    help='comet_ml api-key')
parser.add_argument('--resume', type=str, default='',
                    help='relative path of checkpoint')
args = parser.parse_args()

# image utilities
def rgb2gray(frame):
    return np.dot(frame, [0.2989, 0.5870, 0.1140])

def crop(frame):
    return frame[13:-13, ...]

def resize(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return T.resize(frame, (110, 84))

class DQN(nn.Module):
    def __init__(self, nactions):
        super(DQN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))

        self.linear1 = nn.Sequential(nn.Linear(64*7*7, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(True))
        self.linear2 = nn.Linear(512, nactions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# replay memory
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# env utilities
def reset(env):
    observation = env.reset()
    observation = crop(resize(rgb2gray(observation)))
    
    # convert to byte tensor to conserve memory
    observation = torch.from_numpy(observation).byte()
    return observation

def select_action(model, state, epsilon):
    if random.random() < epsilon:
        action = torch.tensor([[np.random.randint(8)]], 
                              dtype=torch.long)
    else:
        model.eval()
        with torch.no_grad():
            # state is a deque --> convert to single tensor
            # state is stored as byte tensors --> need to convert to float
            state = torch.stack(state).unsqueeze(0).float() / 255
            state = state.to(device, non_blocking=True)
            action = model(state).max(1)[1].view(1, 1)
    return action

def take_step(env, action):
    observation, reward, done, _ = env.step(action.item())
    observation = crop(resize(rgb2gray(observation)))
    observation = torch.from_numpy(observation).byte()
    reward = torch.tensor([reward])
    return observation, reward, done

# Loosly based on: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def optimize(policy_net, target_net, memory, optimizer):
    if len(memory) < HYPERPARAMETERS['batch_size']:
        return
    else:
        policy_net.train()
    
    transitions_ = memory.sample(HYPERPARAMETERS['batch_size'])
    
    # move to device
    transitions = []
    for transition_ in transitions_:
        state, action, next_state, reward = transition_

        # state / next_state is a deque --> convert to single tensor
        # state / next_state are stored as byte tensors --> need to convert to float
        state = torch.stack(state).unsqueeze(0).float() / 255
        state = state.to(device, non_blocking=True)
        if next_state is not None:
            next_state = torch.stack(next_state).unsqueeze(0).float() / 255
            next_state = next_state.to(device, non_blocking=True)
        
        action = action.to(device, non_blocking=True)
        reward = reward.to(device, non_blocking=True)
        
        transitions.append(Transition(state, action, next_state, reward))
     
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                  device=device,
                                  dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                         if s is not None])
    state_batch = torch.cat([s for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.clamp(torch.cat(batch.reward), -1, 1)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_indices = policy_net(non_final_next_states).max(1)[1].view(-1, 1)
    next_state_values = torch.zeros(HYPERPARAMETERS['batch_size'], device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_indices).squeeze().detach()
    
    # Compute the expected Q values
    expected_state_action_values = reward_batch + (next_state_values * HYPERPARAMETERS['gamma'])
    
    # Compute loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using device:',device)

HYPERPARAMETERS = {
        'game': 'VideoPinballDeterministic-v4',
        'total_steps': 50000000,
        'epsilon_start': 1,
        'epsilon_end': 0.1,
        'epsilon_decay_steps': 1000000,
        'policy_update_interval': 4,
        'target_update_interval': 10000,
        'evaluation_interval': 1000000,
        'evaluation_episodes': 100,
        'memory_size': 100000, # set memory_size to 100K for memory reasons (paper: 1M) 
        'gamma': 0.99,
        'lr': 0.00025,
        'batch_size': 32,
        'print_freq': 10000
        }

env = gym.make(HYPERPARAMETERS['game'])

policy_net = DQN(env.action_space.n).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=HYPERPARAMETERS['lr'])

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        num_steps = checkpoint['num_steps'] 
        best_score = checkpoint['best_score']
        epsilon = checkpoint['epsilon']
        last_evaluation = checkpoint['last_evaluation']
        policy_net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.api_key:
            if 'experiment_key' in checkpoint:
                experiment = ExistingExperiment(api_key=args.api_key,
                                                previous_experiment=checkpoint['experiment_key'], 
                                                parse_args=False,
                                                auto_metric_logging=False)
            else:
                experiment = Experiment(api_key=args.api_key, 
                                        project_name='comp767_project', 
                                        parse_args=False,
                                        auto_metric_logging=False)
    else:
        raise Exception
else:
    if args.api_key:
        experiment = Experiment(api_key=args.api_key, 
                                project_name='comp767_project', 
                                parse_args=False,
                                auto_metric_logging=False)
        _, experiment_name = split(dirname(realpath(__file__)))
        experiment.log_other('experiment_name', experiment_name)
        experiment.log_parameters(HYPERPARAMETERS)
    num_steps = 0
    best_score = 0
    last_evaluation = 0
    epsilon = HYPERPARAMETERS['epsilon_start']
print('=> num_steps: {}, best_score: {}, epsilon: {}, last_evaluation: {}\n'.format(num_steps, best_score, epsilon, last_evaluation))

target_net = DQN(env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

start_time = time.time()
memory = ReplayMemory(HYPERPARAMETERS['memory_size'])
while True:
    observation = reset(env)
    queue = deque([observation]*4, maxlen=4)
    while True:
        current_state = list(queue)
        action = select_action(policy_net, current_state, epsilon)
        
        next_observation, reward, done = take_step(env, action)
        queue.appendleft(next_observation)
        next_state = list(queue)
        
        # update replay memory
        if done:
            memory.push(current_state, action, None, reward)
        else:
            memory.push(current_state, action, next_state, reward)
        
        # optimize every 'update_interval' steps
        if num_steps % HYPERPARAMETERS['policy_update_interval'] == 0:
            optimize(policy_net, target_net, memory, optimizer)
        
        # update target network every 'target_update_interval' steps
        if num_steps % HYPERPARAMETERS['target_update_interval'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # print every 'print_freq' steps
        if num_steps % HYPERPARAMETERS['print_freq'] == 0:
            print('steps: {}/{}, time elapsed: {:.1f}'.format(num_steps, HYPERPARAMETERS['total_steps'], time.time()-start_time)) 
        num_steps += 1
        
        # decay epsilon from 'epsilon_start' to 'epsilon_end' over 'epsilon_decay_steps'
        if epsilon > HYPERPARAMETERS['epsilon_end']:
            epsilon = max(0.1, epsilon - ((HYPERPARAMETERS['epsilon_start'] - HYPERPARAMETERS['epsilon_end']) / HYPERPARAMETERS['epsilon_decay_steps']))
        
        # break if episode done
        if done:
            break

    # evaluate every 'evaluation_interval' steps
    if num_steps // HYPERPARAMETERS['evaluation_interval'] > last_evaluation // HYPERPARAMETERS['evaluation_interval']:
        scores = []
        for _ in range(HYPERPARAMETERS['evaluation_episodes']):
            score = 0
            observation = reset(env)
            queue = deque([observation]*4, maxlen=4)
            while True:
                # set epsilon to 0.05 for evaluation
                current_state = list(queue)
                action = select_action(policy_net, current_state, 0.05)
                next_observation, reward, done = take_step(env, action)
                queue.appendleft(next_observation)
                 
                score += reward.item()
                if done:
                    break
            scores.append(score) 
        mean_score = np.mean(scores)
        last_evaluation = num_steps

        if args.api_key:
            with experiment.validate():
                experiment.log_metric('score', mean_score, step=num_steps)
        
        is_best = mean_score > best_score
        best_score = max(mean_score, best_score)
        checkpoint = {
                'num_steps': num_steps,
                'best_score': best_score,
                'epsilon': epsilon,
                'last_evaluation': last_evaluation,
                'state_dict': policy_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
        if args.api_key:
            checkpoint['experiment_key'] = experiment.get_key()
        save_checkpoint(checkpoint, is_best)
        
        # log checkpoints to comet.ml
        if args.api_key:
            experiment.log_asset(file_data='checkpoint.pth.tar', file_name='checkpoint.pth.tar', overwrite=True)
            if is_best:
                experiment.log_asset(file_data='model_best.pth.tar', file_name='model_best.pth.tar', overwrite=True)
        
        print('Evaluate - Mean Score: {}'.format(mean_score))
        
    # stop training after 'total_steps' steps
    if num_steps >= HYPERPARAMETERS['total_steps']:
        break
