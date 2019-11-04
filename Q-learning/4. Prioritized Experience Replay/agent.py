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
    def __init__(self, env, gamma, memory_capacity, hidden_size, lr, batch_size, alpha):
        
        self.env = env
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
 
        self.num_of_states = env.observation_space.shape[0]
        self.num_of_actions = env.action_space.n
        
        self.alpha = alpha
        self.experience_replay_buffer = PrioritizedExperienceReplayBuffer(self.memory_capacity, self.alpha)
        
        # initialize the online (policy) and the target network
        self.online_dqn_network = DQN(self.num_of_states, self.hidden_size, self.num_of_actions)
        self.target_dqn_network = DQN(self.num_of_states, self.hidden_size, self.num_of_actions)
        # the target network will have the same parameter states as the online network
        self.target_dqn_network.load_state_dict(self.online_dqn_network.state_dict())
        
        self.optimizer = optim.Adam(self.online_dqn_network.parameters(), lr = self.lr)
        
    def memorize(self, state, action, new_state, reward, done):
        # this function takes a transition (state, action, new_state, reward, done)
        # and stores it into the experience memory buffer
        self.experience_replay_buffer.push(state, action, new_state, reward, done)
        
    def act(self, state, epsilon):
        # we pick a random probability
        random_prob = random.random()
        # if our probability is below or equal to epsilon
        if random_prob <= epsilon:
            # just pick a random action (exploration)
            action = random.randrange(self.num_of_actions)
        else:
            # don't make a computation graph
            with torch.no_grad():
                state = torch.Tensor(state)
                # depending on our state, get the values for 
                # all possible actions via our policy
                action_from_nn = self.online_dqn_network.forward(state)
                # get the action with the maximum value
                action = torch.max(action_from_nn,0)[1]
                action = action.item()
        return action
        
    def learn(self, beta):
        if len(self.experience_replay_buffer) < self.batch_size:
            return
        else:
            state_batch, action_batch, new_state_batch, reward_batch, done_batch, index_batch, weight_batch = self.experience_replay_buffer.sample(self.batch_size)
            state_batch = torch.tensor(state_batch, dtype=torch.float)
            action_batch = torch.tensor(action_batch, dtype=torch.long)
            new_state_batch = torch.tensor(new_state_batch, dtype=torch.float)
            reward_batch = torch.tensor(reward_batch, dtype=torch.float)
            done_batch = torch.tensor(done_batch, dtype=torch.float)
            index_batch = torch.tensor(index_batch)
            weight_batch = torch.tensor(weight_batch, dtype=torch.float)
            
            y = []

            # estimate the values for the state-actions of the current state batch via our policy network
            # this will have size (batch_size, action_space_shape)
            q_values_online = self.online_dqn_network.forward(state_batch)

            # estimate the values for the state-actions of the next state batch via our policy network
            # this will have size (batch_size, action_space_shape)
            q_values_next_online = self.online_dqn_network.forward(new_state_batch)
            # find the actions with max value from the estimations of the online network
            # this will have size (batch_size)
            max_next_q_values_online = q_values_next_online.max(1)[1]
            # estimate the values for the state-actions of the next state via our target network
            q_values_next_target = self.target_dqn_network.forward(new_state_batch).detach()
            # get the q_values that were estimated from our target networkthat correspond to the 
            # actions with max value as estimated by our policy network
            max_next_q_value_target = q_values_next_target.gather(1, max_next_q_values_online.unsqueeze(1)).squeeze(1)
            
            # the unsqueeze(1) function just takes the action batch that its of dimensions
            # (batch_size,) and just converts it to (batch_size, 1) - squeeze in general
            # adds a 1 at the axis that is specified by the value of the second parameter
            
            # then gather just retrieves all the q_values that we computed
            # that correspond to the indices of the actions in the action_batch
            # (because in every index of the action_batch we only have one action
            #  we have to extract the corresponding q_value of that action from our
            #  q_values predictions).
            
            # the squeeze(1) function just reverts everything to the original
            # dimension that is equal to (batch_size,)
            q_value_online = q_values_online.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            # compute the optimization target (for the states that are final
            # we only want to add the reward)
            y = reward_batch + (self.gamma * max_next_q_value_target) * (1 - done_batch)

            # define the MSE loss between the target and our
            # policy estimated values
            loss = (y - q_value_online).pow(2) * weight_batch
            deltas = loss
            self.experience_replay_buffer.update_priorities(index_batch, deltas)
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            
            # if we need smooth updates we clip the grads between -1 and 1
            #for param in self.online_dqn_network.parameters():
            #    param.grad.data.clamp_(-1,1)
            self.optimizer.step()
                
        return loss
                
            
        
    def update_target_network_params(self):
        self.target_dqn_network.load_state_dict(self.online_dqn_network.state_dict())