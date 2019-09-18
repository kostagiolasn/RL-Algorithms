import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from utils import *
from model import *

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGagent(object):
	def __init__(self, env, gamma = 0.99, tau = 0.001, memory_capacity = 1e6,
				actor_lr = 0.0001, critic_lr = 0.001, batch_size = 64):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.env = env
		# get number of possible states and actions
		self.nb_states = env.obseration_space.shape[0]
		self.nb_actions = env.action_space.shape[0]

		# hyperparameters
		self.gamma = gamma
		self.tau = tau
		self.memory_capacity = memory_capacity
		self.batch_size = batch_size
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr

		# initialize actor and critic networks, as well as their optimizers
		self.actor = Actor(nb_states, nb_actions).to(device)
		self.actor_target = Actor(nb_states, nb_actions).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_lr)

		self.critic = Critic(nb_states, nb_actions).to(device)
		self.critic_target = Critic(nb_states, nb_actions).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.cricit_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)

		# initialize experience replay memory
		self.memory = ReplayMemory(self.memory_capacity)

		# initialize random noisifying process
		self.random_process = OU_Noise(self.nb_actions)

		# Initialize most recent state and action
		self.s_t = None
		self.a_t = None
		self.is_training = True

		self.steps_done = 0

	def select_action(self, s_t):
		# get an action from the actor and then apply noise
		# to it for exploration purposes
		action = to_numpy(self.actor(to_tensor(np.array([s_t])))).squeeze(0)
		action += self.is_training * self.random_process.get_action(s_t)

		# set this action as current action
		self.a_t
		return action

	def optimize_model():
		if len(self.memory) < self.batch_size:
			return
		transitions = memory.sample(self.batch_size)

		batch = Transition(*zip(*transitions))

		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    									batch.next_state)), device = device,
    									dtype = torch.uint8)

	    non_final_next_states = torch.cat([s for s in batch.next_state
	    											if s is not None])
	    state_batch = torch.cat(batch.state)
	    action_batch = torch.cat(batch.action)
	    reward_batch = torch.cat(batch.reward)

	    state_action_values = self.actor(state_batch).gather(1, action_batch)

	    next_state_values = torch.zeros(self.batch_size, device = device)
	    next_state_values[non_final_mask] = self.critic(non_final_next_states).max(1)[0].detach()

	    