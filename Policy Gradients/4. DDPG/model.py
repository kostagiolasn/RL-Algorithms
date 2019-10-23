import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch as T

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
	# The actor is the deterministic policy network
	def __init__(self, nb_states, nb_actions, lr, hidden_size1 = 400, 
		hidden_size2 = 300):
		super(Actor, self).__init__()

		self.init_w = 3e-3
		self.nb_states = nb_states
		self.nb_actions = nb_actions

		#self.bn0 = nn.BatchNorm1d(nb_states)
		self.fc1 = nn.Linear(nb_states, hidden_size1)
		#self.bn1 = nn.BatchNorm1d(hidden_size1)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
		#self.bn2 = nn.BatchNorm1d(hidden_size1)
		self.relu2 = nn.ReLU()
		self.out_fc3 = nn.Linear(hidden_size2, nb_actions)
		self.output_tanh = nn.Tanh()
		self.lr = lr
		self.init_weights()

		self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

		self.to(self.device)

	def init_weights(self):
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		# The paper states: the final weights and biases of the actor
		# network were initialized from a uniform distribution [-3 x 1e-3, 3 x 1e-3]
		self.out_fc3.weight.data.uniform_(-self.init_w, self.init_w)

	def forward(self, states):
		# The paper states that for the deterministic policy network
		# batch normalization was used on the state input and all the
		# layers of the network

		#x = self.bn0(states)
		x = self.fc1(states)
		#x = self.bn1(x)
		x = self.relu1(x)
		x = self.fc2(x)
		#x = self.bn2(x)
		x = self.relu2(x)
		x = self.out_fc3(x)
		# The paper states that the final output layer of the actor
		# is a tanh layer, to bound the actions.
		output = self.output_tanh(x)

		return output

class Critic(nn.Module):
	# The critic is the Q network
	def __init__(self, nb_states, nb_actions, lr,
		hidden_size1 = 400, hidden_size2 = 300):
		super(Critic, self).__init__()

		self.nb_states = nb_states
		self.nb_actions = nb_actions
		self.hidden_size1 = hidden_size1
		self.hidden_size2 = hidden_size2
		self.init_w = 3e-4
		self.lr = lr

		self.fc1 = nn.Linear(nb_states, hidden_size1)
		#self.bn1 = nn.BatchNorm1d(hidden_size1)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size1 + nb_actions, hidden_size2)
		#self.bn2 = nn.BatchNorm1d(hidden_size2)
		self.relu2 = nn.ReLU()
		self.out_fc3 = nn.Linear(hidden_size2, 1)
		self.init_weights()

		self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

		self.to(self.device)

	def init_weights(self):
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		self.out_fc3.weight.data.uniform_(-self.init_w, self.init_w)

	def forward(self, states, actions):

		x = self.fc1(states)
		#x = self.bn1(x)
		x = self.relu1(x)
		# The paper states that the actions were not included
		# until the 2nd hidden layer of the Q network.
		x = torch.cat((x, actions), dim = 1)
		x = self.fc2(x)
		#x = self.bn2(x)
		x = self.relu2(x)
		output = self.out_fc3(x)

		return output



