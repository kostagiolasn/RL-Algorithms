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

class MonteCarloPG_agent(object):
	def __init__(self, env, gamma, hidden_size, lr, batch_size):
        
		self.env = env
		self.hidden_size = hidden_size
		self.lr = lr
		self.batch_size = batch_size
 
		self.num_of_states = env.observation_space.shape[0]
		self.num_of_actions = env.action_space.n
                
        # initialize the online (policy) and the target network
		self.policy_network = PolicyNet(self.num_of_states, self.hidden_size, self.num_of_actions)
        
		self.optimizer = optim.Adam(self.policy_network.parameters(), lr = self.lr)        
        
	def act(self, state):
    	# compute the action distribution based on the current state via the policy net
		action_distribution = self.policy_network.forward(state)

        # pick an action based on that distribution
		action = np.random.choice(self.num_of_actions, p = action_distribution.detach().numpy())
		return action
        
	def learn(self, rewards_batch, states_batch, actions_batch):

		#states_batch = torch.tensor(states_batch, dtype=torch.float)
		states_batch = np.asarray(states_batch)
		actions_batch = torch.tensor(actions_batch, dtype=torch.long)
		rewards_batch = torch.tensor(rewards_batch, dtype=torch.float)
        
        # compute the log of the probabilities that the policy outputs for each state
		log_probs = torch.log(self.policy_network(states_batch))
        # pick those log probabilities that correspond to the actions that were selected
		selected_log_probs = rewards_batch * log_probs[np.arange(len(actions_batch)), actions_batch]
        # compute the monte-carlo estimate by averaging the losses and then form the optimization
        # criterion, which will be the negative log probs.
		loss = -selected_log_probs.mean()
		self.optimizer.zero_grad()
		loss.backward()
        
        # if we need smooth updates we clip the grads between -1 and 1
        #for param in self.online_dqn_network.parameters():
        #    param.grad.data.clamp_(-1,1)
		self.optimizer.step()
                
		return loss