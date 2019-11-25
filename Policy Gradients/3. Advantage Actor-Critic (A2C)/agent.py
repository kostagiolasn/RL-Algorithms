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

class A2C_agent(object):
	def __init__(self, env, actor_hidden_size, actor_lr, actor_batch_size, 
		critic_gamma, mem_size, critic_hidden_size, critic_lr, critic_batch_size):
        
		self.env = env
		self.actor_hidden_size = actor_hidden_size
		self.actor_lr = actor_lr
		self.actor_batch_size = actor_batch_size

		self.critic_hidden_size = critic_hidden_size
		self.critic_lr = critic_lr
		self.critic_batch_size = critic_batch_size
		self.critic_gamma = critic_gamma

		self.mem_size = mem_size
 
		self.num_of_states = env.observation_space.shape[0]
		self.num_of_actions = env.action_space.n

		self.experience_replay_buffer = ReplayBuffer(self.mem_size)
                
        # initialize the Actor network (policy)
		self.actor_network = ActorNet(self.num_of_states, self.actor_hidden_size, self.num_of_actions)
        
		self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr = self.actor_lr)  

		# initialize the Critic network (v-learning)
		# The difference between the critic in A2C (here) and the 
	# critic int he "vanilla" Actor-Critic version is that the
	# critic in A2C models the value function, hence it needs
	# to only output the value of each state and not the Q-value
	# for each (state, action) pair. Therefore, the output size
	# here needs to be a scalar.
		self.critic_network = CriticNet(self.num_of_states, self.critic_hidden_size, 1)
        
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr = self.critic_lr)        
        
	def act(self, state):
    	# compute the action distribution based on the current state via the policy net
		action_distribution = self.actor_network.forward(state)

        # pick an action based on that distribution
		action = np.random.choice(self.num_of_actions, p = action_distribution.detach().numpy())
		return action
		
	def memorize(self, state, action, new_state, reward, done):
        # this function takes a transition (state, action, new_state, reward, done)
        # and stores it into the experience memory buffer
		self.experience_replay_buffer.push(state, action, new_state, reward, done)

	def learn(self, rewards_batch, states_batch, actions_batch, new_states_batch, new_actions_batch):

		#states_batch = torch.tensor(states_batch, dtype=torch.float)
		states_batch = np.asarray(states_batch)
		actions_batch = torch.tensor(actions_batch, dtype=torch.long)
		rewards_batch = torch.tensor(rewards_batch, dtype=torch.float)
		new_states_batch = np.asarray(states_batch)
		new_actions_batch = torch.tensor(actions_batch, dtype=torch.long)
		V_batch = []
		V_prime_batch = []

		for state, new_state, new_action in zip(states_batch,\
			new_states_batch, new_actions_batch):
			state = torch.Tensor(state)

			v_value = self.critic_network.forward(state)
			# get q-value for specific action
			#Q = q_values.gather(-1, action)
			V_batch.append(v_value)

			new_state = torch.Tensor(new_state)
			v_prime_value = self.critic_network.forward(new_state)
			#V_prime = q_prime_values.gather(-1, new_action)
			V_prime_batch.append(v_prime_value)
        
        # compute the log of the probabilities that the policy outputs for each state
		log_probs = torch.log(self.actor_network(states_batch))
        # pick those log probabilities that correspond to the actions that were selected
		selected_log_probs = rewards_batch * log_probs[np.arange(len(actions_batch)), actions_batch]
        # compute the monte-carlo estimate by averaging the losses and then form the optimization
        # criterion, which will be the negative log probs.
		actor_loss = -selected_log_probs.mean()
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
        
        # if we need smooth updates we clip the grads between -1 and 1
        #for param in self.online_dqn_network.parameters():
        #    param.grad.data.clamp_(-1,1)
		self.actor_optimizer.step()

		# Compute TD error for V network
		V_prime_batch = torch.stack(V_prime_batch)
		V_batch = torch.stack(V_batch)
		# A(s, a) = r_prime + gamma * V_prime - V
		advantage = rewards_batch + self.critic_gamma * V_prime_batch - V_batch
		#print(deltas)

		critic_loss = (V_batch - (rewards_batch + self.critic_gamma * V_prime_batch)).pow(2).mean()
		#print(critic_loss)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()


		#return loss