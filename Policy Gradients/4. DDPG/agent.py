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

class DDPG_agent(object):
	def __init__(self, env, hidden_size1, hidden_size2, actor_lr, critic_gamma,\
		critic_lr, batch_size, mem_size, tau):
        
		self.env = env
		self.hidden_size1 = hidden_size1
		self.hidden_size2 = hidden_size2
		self.actor_lr = actor_lr

		self.critic_gamma = critic_gamma
		self.critic_lr = critic_lr

		self.batch_size = batch_size
		self.critic_gamma = critic_gamma

		self.mem_size = mem_size
		self.tau = tau
 
		self.num_of_states = env.observation_space.shape[0]
		self.num_of_actions = env.action_space.shape[0]

		self.experience_replay_buffer = ReplayBuffer(self.mem_size)
                
        # initialize the Actor network (policy)
		self.actor_network = ActorNet(self.num_of_states, self.hidden_size1, self.hidden_size2, self.num_of_actions)        
		self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr = self.actor_lr)  
		self.target_actor_network = ActorNet(self.num_of_states, self.hidden_size1, self.hidden_size2, self.num_of_actions) 
		self.target_actor_network.load_state_dict(self.actor_network.state_dict())

		self.critic_network = CriticNet(self.num_of_states, self.hidden_size1, self.hidden_size2, self.num_of_actions)
		self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr = self.critic_lr) 
		self.target_critic_network = CriticNet(self.num_of_states, self.hidden_size1, self.hidden_size2, self.num_of_actions)
		self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        
	def act(self, state):
    	# compute the enxt action based on the current state via the policy net
		self.actor_network.eval()
		state = Variable(torch.from_numpy(state).type(torch.FloatTensor))
		action = self.actor_network.forward(state)
		self.actor_network.train()
		return action
		
	def memorize(self, state, action, new_state, reward, done):
        # this function takes a transition (state, action, new_state, reward, done)
        # and stores it into the experience memory buffer
		self.experience_replay_buffer.push(state, action, new_state, reward, done)

	def learn(self):
		if len(self.experience_replay_buffer) < self.batch_size:
			return
		else:
			# Sample a random minibatch of transitions from replay memory
			state_batch, action_batch, new_state_batch, reward_batch, done_batch = self.experience_replay_buffer.sample(self.batch_size)

			state_batch = torch.FloatTensor(state_batch)
			action_batch = torch.FloatTensor(action_batch)
			new_state_batch = torch.FloatTensor(new_state_batch)
			reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
			done_batch = torch.FloatTensor(np.float32(done_batch)).unsqueeze(1)

			self.target_actor_network.eval()
			self.target_critic_network.eval()
			self.critic_network.eval()

			# get the next action using the target policy (actor) network
			new_action_batch = self.target_actor_network.forward(new_state_batch)
			# get the q values for the next state using the  target value (critic) network
			new_state_q_values = self.target_critic_network.forward(new_state_batch, new_action_batch)
			# set y
			y = reward_batch + self.critic_gamma * new_state_q_values * (1 - done_batch)

			# compute the values for the current state
			state_q_values = self.critic_network.forward(state_batch, action_batch)
			q_values = reward_batch + self.critic_gamma * state_q_values * (1 - done_batch)
			
			# Critic update: classic value network update based on the squared TD error
			self.critic_network.train()
			critic_loss = (y - q_values).pow(2).mean()
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_network.eval()

			# Actor update: this update is just based on the sampled policy gradient
			mu = self.actor_network.forward(state_batch)
			self.actor_network.train()
			actor_loss = - self.critic_network.forward(state_batch, mu).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_network.eval()

			# perform the update for the targets
			self.soft_update_target_nets(self.tau)

	def soft_update_target_nets(self, tau):
		for target_param, param in zip(self.target_critic_network.parameters(), \
			self.critic_network.parameters()):
			target_param.data.copy_(
				target_param.data * (1.0 - self.tau) + param.data * self.tau
			)

		for target_param, param in zip(self.target_actor_network.parameters(), \
			self.actor_network.parameters()):
			target_param.data.copy_(
				target_param.data * (1.0 - self.tau) + param.data * self.tau
			)



		