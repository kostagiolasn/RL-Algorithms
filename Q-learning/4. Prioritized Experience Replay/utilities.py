#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:21:21 2019

@author: Nikos
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import math

class PrioritizedExperienceReplayBuffer(object):
    def __init__(self, capacity, alpha = 0.6):
        self.experience_memory = []
        self.capacity = capacity
        self.offset = 0
        self.priorities = np.zeros((capacity,), dtype = np.float32)
        self.t = 0
        self.alpha = alpha

    def push(self, state, action, new_state, reward, done):

        max_priority = self.priorities.max() if self.offset > 0 else 1.0

        transition = (state, action, new_state, reward, done)
        
        if self.offset >= len(self.experience_memory):
            self.experience_memory.append(transition)
        else:
            self.experience_memory[self.offset] = transition

        # the memory that will be inserted will have priority equal to the
        # max priority found in memory
        self.priorities[self.offset] = max_priority
        
        # if the experience memory is full then proceed
        # to overwrite old experiences. in general the memory
        # input/output works in a FIFO sense
        self.offset = (self.offset + 1) % self.capacity
    
    def sample(self, batch_size, beta = 0.4):
        if len(self.experience_memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.offset]

        sample_probabilities = priorities ** self.alpha
        sample_probabilities /= sum(sample_probabilities)

        # sample a batch of transitions according to the sample probabilities
        transition_indices = np.random.choice(len(self.experience_memory), batch_size, p = sample_probabilities)
        sample_transitions = [self.experience_memory[transition_index] for \
        transition_index in transition_indices]

        weights = (len(self.experience_memory) * \
            sample_probabilities[transition_indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype = np.float32)

        batch = list(zip(*sample_transitions))
        batch_states = batch[0]
        batch_actions = batch[1]
        batch_new_states = batch[2]
        batch_rewards = batch[3]
        batch_dones = batch[4]
        batch_indices = transition_indices
        batch_weights = weights

        return batch_states, batch_actions, batch_new_states, batch_rewards,\
            batch_dones, batch_indices, batch_weights

    def update_priorities(self, batch_indices, batch_deltas):
        batch_indices = batch_indices.data.cpu().numpy()
        batch_deltas = batch_deltas.data.cpu().numpy()
        for index, delta in zip(batch_indices, batch_deltas):
            self.priorities[index] = np.absolute(delta)
    
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