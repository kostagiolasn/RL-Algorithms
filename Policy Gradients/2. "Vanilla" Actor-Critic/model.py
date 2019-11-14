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
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
                
    def forward(self, x):
    	x = Variable(torch.from_numpy(x).type(torch.FloatTensor))
    	x = F.relu(self.hidden1(x))
    	# The softmax is put here because the policy
    	# is supposed to model a distribution over actions
    	# given states.
    	action_scores = self.output(x)
    	out = F.softmax(action_scores, dim = -1)
    	return out

class CriticNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticNet, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
                
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x