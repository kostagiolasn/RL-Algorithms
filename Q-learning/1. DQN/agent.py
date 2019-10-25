#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:21:32 2019

@author: Nikos
"""

import torch
import torch.optim as optim
from utilities import *
from model import *
from torch.autograd import Variable
import numpy as np
import random

class DQN_agent(object):
    def __init__(self, env, gamma, memory_capacity, hidden_size, lr, batch_size):
        
        self.env = env
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
 
        self.num_of_states = env.observation_space.shape[0]
        self.num_of_actions = env.action_space.n
        
        self.experience_replay_buffer = ReplayBuffer(self.memory_capacity)
        
        self.online_dqn_network = DQN(self.num_of_states, self.hidden_size, self.num_of_actions)
        self.target_dqn_network = DQN(self.num_of_states, self.hidden_size, self.num_of_actions)
        self.target_dqn_network.load_state_dict(self.online_dqn_network.state_dict())
        
        self.optimizer = optim.Adam(self.online_dqn_network.parameters(), lr = self.lr)
        
    def memorize(self, state, action, new_state, reward, done):
        self.experience_replay_buffer.push(state, action, new_state, reward, done)
        
    def act(self, state, epsilon):
        random_prob = random.random()
        if random_prob <= epsilon:
            action = random.randrange(self.num_of_actions)
        else:
            with torch.no_grad():
                state = torch.Tensor(state)
                action_from_nn = self.online_dqn_network.forward(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()
        return action
        
    def learn(self):
        if len(self.experience_replay_buffer) < self.batch_size:
            return
        else:
            state_batch, action_batch, new_state_batch, reward_batch, done_batch = self.experience_replay_buffer.sample(self.batch_size)
            state_batch = torch.tensor(state_batch, dtype=torch.float)
            action_batch = torch.tensor(action_batch, dtype=torch.long)
            new_state_batch = torch.tensor(new_state_batch, dtype=torch.float)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float)
            done_batch = torch.tensor(done_batch, dtype=torch.float)
            
            y = []
            q_values_online = self.online_dqn_network.forward(state_batch)
            next_q_values_target = self.target_dqn_network.forward(new_state_batch).detach()
                    
            q_value_online = q_values_online.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            max_next_q_value_target = next_q_values_target.max(1)[0]
            y = reward_batch + (self.gamma * max_next_q_value_target) * (1 - done_batch)
                
            loss = (y - q_value_online).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            
            # if we need smooth updates we clip the grads between -1 and 1
            for param in online_dqn_network.parameters():
                param.grad.data.clamp_(-1,1)
            self.optimizer.step()
                
        return loss
                
            
        
    def update_target_network_params(self):
        self.target_dqn_network.load_state_dict(self.online_dqn_network.state_dict())