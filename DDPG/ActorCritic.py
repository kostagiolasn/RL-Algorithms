import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class Critic(nn.Module):
	# The critic is the Q network
	def __init__(self, hidden_layer_size1 = 400, hidden_layer_size2 = 300,
		 input_dim, output_q_value, action_dim):
		super(Critic, self).__init__()

		self.input_dim = input_dim
		self.action_dim = action_dim
		self.hidden_layer_size1 = hidden_layer_size1
		self.hidden_layer_size2 = hidden_layer_size2
		self.init_w = 3e-3

		self.bn1 = nn.BatchNorm2d(input_dim)
		self.linear1 = nn.Linear(input_dim, hidden_layer_size1)
		self.bn2 = nn.BatchNorm2d(hidden_layer_size1)
		self.relu1 = nn.ReLU(inplace = True)
		self.linear2 = nn.Linear(hidden_layer_size1, hidden_layer_size2)
		self.bn3 = nn.BatchNorm2d(hidden_layer_size2)
		self.relu2 = nn.ReLU(inplace = True)
		self.output = nn.Linear(hidden_layer_size2, output_size)
		self.softmax = nn.Softmax()

	def init_weights(self, init_w):
        self.linear1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.linear2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

	def forward(self, states, actions):
		x = self.linear1(states)
		x = self.relu1(x)
		x = torch.cat((x, actions), dim = 1)
		x = self.linear2(x)
		x = self.relu2(x)
		output = self.output(x)

		return output

class Actor(nn.Module):
	# The actor is the deterministic policy network
	def __init__(self, input_dim, hidden_layer_size1 = 400, 
		hidden_layer_size2 = 300, output_q_value, learning_rate = 1e-4):
		super(Critic, self).__init__()

		self.init_w = 3e-3

		self.linear1 = nn.Linear(input_size, hidden_layer_size1)
		self.bn1 = nn.BatchNorm2d(hidden_layer_size1)
		self.relu1 = nn.ReLU(inplace = True)
		x = 
		self.linear2 = nn.Linear(hidden_layer_size1, hidden_layer_size2)
		self.relu2 = nn.ReLU(inplace = True)
		self.output = nn.Linear(hidden_layer_size2, output_size)

	def init_weights(self, init_w):
		self.dense_1.weight.data = fanin_init(self.linear1.weight.data.size())
        self.dense_2.weight.data = fanin_init(self.linear2.weight.data.size())
        self.output.weight.data.uniform_(-init_w, init_w)

	def forward(self, states, actions):
		# The paper states that actions were not included until the 2nd
		# layer of the Q network.
		x = F.relu(self.bn1(self.linear1(state)))
		x = F.relu(self.linear2(x))
		# The paper states that the final output layer of the actor
		# is a tanh layer, to bound the actions.
		output = torch.tanh(self.output(x))

		return output

