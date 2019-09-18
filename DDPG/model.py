import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Actor(nn.Module):
	# The actor is the deterministic policy network
	def __init__(self, nb_states, nb_actions, hidden_size1 = 400, 
		hidden_size2 = 300):
		super(Actor, self).__init__()

		self.init_w = 3e-3
		self.nb_states = nb_states
		self.nb_actions = nb_actions

		self.bn0 = nn.BatchNorm(hidden_size1)
		self.fc1 = nn.Linear(nb_states, hidden_size1)
		self.bn1 = nn.BatchNorm(hidden_size1)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
		self.bn2 = nn.BatchNorm(hidden_size1)
		self.relu2 = nn.ReLU()
		self.out_fc3 = nn.Linear(hidden_size2, nb_actions)
		self.output_tanh = nn.Tanh()
		self.init_weights()

	def init_weights(self):
		self.fc1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.linear2.weight.data.size())
        # The paper states: the final weights and biases of the actor
        # network were initialized from a uniform distribution [-3 x 1e-3, 3 x 1e-3]
        self.out_fc3.weight.data.uniform_(-self.init_w, self.init_w)

	def forward(self, states, actions):
		# The paper states that for the deterministic policy network
		# batch normalization was used on the state input and all the
		# layers of the network
		x = self.bn0(states)
		x = self.fc1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		x = self.out_fc3(x)
		# The paper states that the final output layer of the actor
		# is a tanh layer, to bound the actions.
		output = self.output_tanh(x)

		return output

class Critic(nn.Module):
	# The critic is the Q network
	def __init__(self, hidden_size1 = 400, hidden_size2 = 300,
		 nb_states, nb_actions, action_dim):
		super(Critic, self).__init__()

		self.nb_states = nb_states
		self.nb_actions = nb_actions
		self.hidden_size1 = hidden_size1
		self.hidden_size2 = hidden_size2
		self.init_w = 3e-4

		self.fc1 = nn.Linear(input_dim, hidden_size1)
		self.bn1 = nn.BatchNorm(hidden_size1)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size1 + nb_actions, hidden_size2)
		self.bn2 = nn.BatchNorm(hidden_size2)
		self.out_fc3 = nn.Linear(hidden_size2, output_size)
		self.init_weights()

	def init_weights(self):
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.output.weight.data.uniform_(-self.init_w, self.init_w)

	def forward(self, states, actions):
		x = self.linear1(states)
		x = self.bn1(x)
		x = self.relu1(x)
		# The paper states that the actions were not included
		# until the 2nd hidden layer of the Q network.
		x = torch.cat((x, actions), dim = 1)
		x = self.linear2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		output = self.out_fc3(x)

		return output



