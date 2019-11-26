#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:20:51 2019

@author: Nikos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ActorNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_actions, init_w = 3e-3):
        super(ActorNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output = nn.Linear(hidden_size2, num_actions)

        self.output.weight.data.uniform_(-init_w, init_w)
        self.output.bias.data.uniform_(-init_w, init_w)
                
    def forward(self, state):
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action_scores = self.output(x)
        out = F.tanh(action_scores)
        return out

class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_actions, init_w = 3e-3):
        super(CriticNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1 + num_actions, hidden_size2)
        self.output = nn.Linear(hidden_size2, 1)
        self.output.weight.data.uniform_(-init_w, init_w)
        self.output.bias.data.uniform_(-init_w, init_w)
                
    def forward(self, state, action):
        x = F.relu(self.hidden1(state))
        x = torch.cat([x, action], 1)
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x