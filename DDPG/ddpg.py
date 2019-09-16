import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from NormalizedEnv import *
from Memory import *
from ActorCritic import *
from OU_noise import *

class DDPGagent:
	def __init__(self, env, gamma = 0.99, tau = 0.001, memory_capacity = 1e6,
				actor_lr = 0.0001, critic_lr = 0.001):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.env = env
		self.obs_dim = env.obseration_space.shape[0]
		self.action_dim = env.action_space.shape[0]

		# hyperparameters
		self.gamma = gamma
		self.tau = tau

		# initialize actor and critic networks
		

	def get_action():

	def update():