#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:21:44 2019

@author: Nikos
"""

import math
import gym
from utilities import *
from model import *
from agent import *
import random
import numpy as np

def main():
    
	random.seed(1234)
	np.random.seed(1234)
    
    ### Policy Gradient HYPERPARAMETERS ###
	pg_batch_size = 32
	pg_total_episodes = 10000
	pg_hidden_size = 128
	pg_gamma = 0.99
	pg_lr = 0.01
    #######################

    ### DQN HYPERPARAMETERS ###
	dqn_batch_size = 128
	mem_size = 10000
	dqn_total_episodes = 10	
	dqn_gamma = 0.99
	dqn_hidden_size = 128
	dqn_lr = 0.02
    ###########################
    	
	score_history = []
        
	env_name = "CartPole-v0"
	env = gym.make(env_name)
    
	A2C_agent = A2C_agent(env, pg_hidden_size, pg_lr, pg_batch_size,\
		dqn_gamma, mem_size, dqn_hidden_size, dqn_lr, dqn_batch_size)
    
	state = env.reset()
    
	for episode_num in range(0, pg_total_episodes):
        
		done = False
		score = 0

		rewards = []
		states = []
		actions = []
		new_states = []
		new_actions = []

		while not done:
			# pick an action
			action = A2C_agent.act(state)            
			# act and get feedback from the environment
			new_state, reward, done, info = env.step(action)

			# add the transition to the experience buffer
			A2C_agent.memorize(state, action, new_state, reward, done)

			rewards.append(reward)
			states.append(state)
			actions.append(action)

			new_action = A2C_agent.act(new_state)

			new_states.append(new_state)
			new_actions.append(new_action)

			score += reward
			# the state now will be the new state
			state = new_state

		current_pg_batch_size = 0
		rewards_batch = []
		states_batch = []
		actions_batch = []
		new_states_batch = []
		new_actions_batch = []

        # now that we are done we can compute the reward estimate for each state
		t = 0
		for reward, state, action in zip(rewards, states, actions):
			if current_pg_batch_size == pg_batch_size:
        		# compute loss and backprop
				A2C_agent.learn(rewards_batch, states_batch, actions_batch,\
				 new_states_batch, new_actions_batch)

        		# and reinitialize batch information
				current_pg_batch_size = 0
				rewards_batch = []
				states_batch = []
				actions_batch = []
				new_states_batch = []
				new_actions_batch = []

			# compute discounted reward
			discounted_reward = compute_discounted_reward(rewards[t:], pg_gamma)

			t = t + 1

			# add it to the batch
			rewards_batch.extend([discounted_reward.tolist()])

			# add the state and action to their respective batches
			states_batch.extend([state])
			actions_batch.extend([action])

			current_pg_batch_size += 1

        
		state = env.reset()
            
		score_history.append(score)
        
		print('episode ', episode_num, 'reward %.2f' % score,
          'trailing 10 games avg %.3f' % np.mean(score_history[-10:]))
        
	x = [i + 1 for i in range(pg_total_episodes)]
	filename = 'CartPole-v0.png'
	plotLearning(x, score_history, filename)
    
if __name__ == "__main__":
	main()