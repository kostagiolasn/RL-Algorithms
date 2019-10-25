#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:21:21 2019

@author: Nikos
"""

import random
import numpy as np
import matplotlib.pyplot as plt

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.experience_memory = []
        self.capacity = capacity
        self.offset = 0
        
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        
        if self.offset >= len(self.experience_memory):
            self.experience_memory.append(transition)
        else:
            self.experience_memory[self.offset] = transition
        
        self.offset = (self.offset + 1) % self.capacity
    
    def sample(self, batch_size):
        return zip(*random.sample(self.experience_memory, batch_size))
    
    def __len__(self):
        return len(self.experience_memory)
    
def plotLearning(x, scores, filename):
	window = 10
	smoothed_scores = [np.mean(scores[i-window:i+1]) if i > window 
                    else np.mean(scores[:i+1]) for i in range(len(scores))]

	plt.figure(figsize=(12,8))
	plt.plot(x, scores)
	plt.plot(x, smoothed_scores)
	plt.ylabel('Total Scores')
	plt.xlabel('Episodes')
	plt.show()
	plt.savefig(filename)