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
    batch_size = 128
    mem_size = 10000
    total_episodes = 10000
    gamma = 0.99
    epsilon_start = 0.9
    epsilon_finish = 0.05
    epsilon_decay = 200
    hidden_size = 128
    lr = 0.02
    C = 10
    #######################
    
    score_history = []
        
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    
    dqn_agent = DQN_agent(env, gamma, mem_size, hidden_size, lr, batch_size)
    
    state = env.reset()
    
    for episode_num in range(0, total_episodes):
        
        if C % 100:
            dqn_agent.update_target_network_params()
        
        current_epsilon_value = epsilon_finish + (epsilon_start - epsilon_finish) *\
            math.exp(-1 * episode_num / epsilon_decay)
        
        done = False
        score = 0
        
        while not done:
            action = dqn_agent.act(state, current_epsilon_value)
            new_state, reward, done, info = env.step(action)
            dqn_agent.memorize(state, action, new_state, reward, done)
            dqn_agent.learn()
            score += reward
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