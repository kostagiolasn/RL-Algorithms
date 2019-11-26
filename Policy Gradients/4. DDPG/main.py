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
    
	total_episodes = 10000
	batch_size = 64
	mem_size = 10000
	hidden_size1 = 400
	hidden_size2 = 300
	tau = 0.001
	max_steps = 500

    ### Actor HYPERPARAMETERS ###
	actor_lr = 0.0001
    #############################

    ### Critic HYPERPARAMETERS ###
	critic_gamma = 0.99
	critic_lr = 0.001
    ##############################

    ### Ornstein-Uhlenbeck process HYPERAPRAMETERS ###
	theta = 0.15
	sigma = 0.2
    ##################################################
    	
	score_history = []
        
	env_name = "Walker2d-v2"
	env = gym.make(env_name)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	exploration_noise = OUNoise(env.action_space, sigma, theta)
    
	ddpg_agent = DDPG_agent(env, hidden_size1, hidden_size2, actor_lr, critic_gamma,\
		critic_lr, batch_size, mem_size, tau)
    
	state = env.reset()
    
	for episode_num in range(0, total_episodes):
        
		done = False
		score = 0

		rewards = []
		states = []
		actions = []
		new_states = []
		new_actions = []

		for step in range(max_steps):
			# pick an action
			action = ddpg_agent.act(state)
			noisified_action = exploration_noise.noisify_action(action, step)          
			# act and get feedback from the environment
			new_state, reward, done, info = env.step(noisified_action)

			# add the transition to the experience buffer
			ddpg_agent.memorize(state, noisified_action, new_state, reward, done)

			rewards.append(reward)
			states.append(state)
			actions.append(noisified_action)

			new_action = ddpg_agent.act(new_state)

			new_states.append(new_state)
			new_actions.append(new_action)

			score += reward
			# the state now will be the new state
			state = new_state

			ddpg_agent.learn()

        
		state = env.reset()
		exploration_noise.reset()
            
		score_history.append(score)
        
		print('episode ', episode_num, 'reward %.2f' % score,
          'trailing 10 games avg %.3f' % np.mean(score_history[-10:]))
        
	x = [i + 1 for i in range(pg_total_episodes)]
	filename = 'CartPole-v0.png'
	plotLearning(x, score_history, filename)
    
if __name__ == "__main__":
	main()