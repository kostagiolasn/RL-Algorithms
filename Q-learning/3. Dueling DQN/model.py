#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:20:51 2019

@author: Nikos
"""

import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        # This is the part that computes the advantage
        # for the state-action (s,a)
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.advantage_output = nn.Linear(hidden_size, output_size)

        # This is the part that computes the value
        # for the state s, which is just a scalar
        self.hidden3 = nn.Linear(input_size, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        self.value_output = nn.Linear(hidden_size, 1)
                
    def forward(self, x):
        # We compute the advantage function
        x1 = F.relu(self.hidden1(x))
        x1 = F.relu(self.hidden2(x1))
        advantage = self.advantage_output(x1)

        # We compute the value function
        x2 = F.relu(self.hidden3(x))
        x2 = F.relu(self.hidden4(x2))
        value = self.value_output(x2)

        # Q(s,a) = V(s) + A(s,a) but we also subtract the
        # mean of the advantage function to enforce that the
        # mean of the A(s,a) will be zero.
        q = value + advantage - advantage.mean()

        return q