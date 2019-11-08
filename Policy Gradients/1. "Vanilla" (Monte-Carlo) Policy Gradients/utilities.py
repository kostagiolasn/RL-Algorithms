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

def compute_discounted_reward(rewards, gamma = 0.99):
    r = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])

    # compute the cumulative sum
    r = r[::-1].cumsum()[::-1]

    # return its normalized value
    return r - r.mean()
    
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