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
    
    ### HYPERPARAMETERS ###
	batch_size = 32
	total_episodes = 10000
	hidden_size = 128
	gamma = 0.99
	lr = 0.01
    #######################
    
	score_history = []
        
	env_name = "CartPole-v0"
	env = gym.make(env_name)
    
	monte_carlo_pg_agent = MonteCarloPG_agent(env, gamma, hidden_size, lr, batch_size)
    
	state = env.reset()
    
	for episode_num in range(0, total_episodes):
        
		done = False
		score = 0

		rewards = []
		states = []
		actions = []

		while not done:
			# pick an action
			action = monte_carlo_pg_agent.act(state)            
			# act and get feedback from the environment
			new_state, reward, done, info = env.step(action)

			rewards.append(reward)
			states.append(state)
			actions.append(action)

			score += reward
			# the state now will be the new state
			state = new_state

		current_batch_size = 0
		rewards_batch = []
		states_batch = []
		actions_batch = []

        # now that we are done we can compute the reward estimate for each state
		t = 0
		for reward, state, action in zip(rewards, states, actions):
			if current_batch_size == batch_size:
        		# compute loss and backprop
				monte_carlo_pg_agent.learn(rewards_batch, states_batch, actions_batch)

        		# and reinitialize batch information
				current_batch_size = 0
				batch_rewards = []
				batch_states = []
				batch_actions = []

			# compute discounted reward
			discounted_reward = compute_discounted_reward(rewards[t:], gamma)

			t = t + 1

			# add it to the batch
			rewards_batch.extend([discounted_reward.tolist()])

			# add the state and action to their respective batches
			states_batch.extend([state])
			actions_batch.extend([action])

			current_batch_size += 1

        
		state = env.reset()
            
		score_history.append(score)
        
		print('episode ', episode_num, 'reward %.2f' % score,
          'trailing 10 games avg %.3f' % np.mean(score_history[-10:]))
        
	x = [i + 1 for i in range(total_episodes)]
	filename = 'CartPole-v0.png'
	plotLearning(x, score_history, filename)
    
if __name__ == "__main__":
	main()