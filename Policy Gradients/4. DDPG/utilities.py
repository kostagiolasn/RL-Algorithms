#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:21:21 2019

@author: Nikos
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math
import gym
import torch as T


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

class OUNoise(object):
    def __init__(self, action_space, theta=0.15, max_sigma=0.3, min_sigma=0.3, mu = 0.0, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def noisify_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        #ou_state = T.tensor(ou_state, dtype=T.float)
        action = action.detach().numpy()
        return np.clip(action + ou_state, self.low, self.high)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.experience_memory = []
        self.capacity = capacity
        self.offset = 0
        
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.offset >= len(self.experience_memory):
            self.experience_memory.append(transition)
        else:
            self.experience_memory[self.offset] = transition
        
        # if the experience memory is full then proceed
        # to overwrite old experiences. in general the memory
        # input/output works in a FIFO sense
        self.offset = (self.offset + 1) % self.capacity
    
    def sample(self, batch_size):
        # sample random batches of transitions
        return zip(*random.sample(self.experience_memory, batch_size))
    
    def __len__(self):
        return len(self.experience_memory)
    
def plotLearning(x, scores, filename):
    window = 10
    smoothed_scores = [np.mean(scores[i-window:i+1]) if i > window 
                    else np.mean(scores[:i+1]) for i in range(len(scores))]

    plt.figure(figsize=(12,8))
    plt.plot(x, scores)
    plt.plot(x, smoothed_scores)
    plt.ylabel('Total Scores')
    plt.xlabel('Episodes')
    plt.show()
    plt.savefig(filename)

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.experience_memory = []
        self.capacity = capacity
        self.offset = 0
        
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.offset >= len(self.experience_memory):
            self.experience_memory.append(transition)
        else:
            self.experience_memory[self.offset] = transition
        
        # if the experience memory is full then proceed
        # to overwrite old experiences. in general the memory
        # input/output works in a FIFO sense
        self.offset = (self.offset + 1) % self.capacity
    
    def sample(self, batch_size):
        # sample random batches of transitions
        return zip(*random.sample(self.experience_memory, batch_size))
    
    def __len__(self):
        return len(self.experience_memory)