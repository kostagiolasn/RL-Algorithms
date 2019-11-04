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
    mem_size = 10000
    total_episodes = 10000
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_finish = 0.01
    epsilon_decay = 500
    hidden_size = 128
    lr = 0.0001
    C = 10

    # beta: bias annealing factor
    beta_start = 0.4
    beta_finish = 1.0
    beta_episodes = 1000
    # alpha: priority
    alpha = 0.6
    #######################
    
    score_history = []
        
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    
    dqn_agent = DQN_agent(env, gamma, mem_size, hidden_size, lr, batch_size, alpha)
    
    state = env.reset()
    
    for episode_num in range(0, total_episodes):
        
        # every C epochs copy the parameters of the
        # online network to those of the target network
        if episode_num % C:
            dqn_agent.update_target_network_params()
        
        # compute the epsilon for that specific epoch
        current_epsilon_value = epsilon_finish + (epsilon_start - epsilon_finish) *\
            math.exp(-1 * episode_num / epsilon_decay)

        # compute the value of beta for the specific episode.
        # beta will be annealed for the first 1000 episodes
        # and then it will have a constant value equal to 1.0
        current_beta = min(1.0, beta_start + episode_num *\
         (1.0 - beta_start) / beta_episodes)
        
        done = False
        score = 0
        
        while not done:
            # pick an action depending on the current state
            action = dqn_agent.act(state, current_epsilon_value)
            # act and get feedback from the environment
            new_state, reward, done, info = env.step(action)
            # add the transition to the experience buffer
            dqn_agent.memorize(state, action, new_state, reward, done)
            # update the network parameters
            dqn_agent.learn(current_beta)
            score += reward
            # the state now will be the new state
            state = new_state
        
        state = env.reset()
            
        score_history.append(score)
        
        print('episode ', episode_num, 'reward %.2f' % score,
          'trailing 10 games avg %.3f' % np.mean(score_history[-10:]))
        
    x = [i + 1 for i in range(total_episodes)]
    filename = 'CartPole-v0.png'
    plotLearning(x, score_history, filename)
    
if __name__ == "__main__":
    main()