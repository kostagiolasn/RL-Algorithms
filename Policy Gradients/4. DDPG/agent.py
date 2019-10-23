import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch as T
import gym
from utils import *
from model import *

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGagent(object):
	def __init__(self, env, input_dims, nb_actions, gamma = 0.99, tau = 0.001, memory_capacity = 1e6,
				actor_lr = 0.0001, critic_lr = 0.001, batch_size = 64):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.env = env
		# get number of possible states and actions
		self.nb_states = env.observation_space.shape[0]
		self.nb_actions = env.action_space.shape[0]

		# hyperparameters
		self.gamma = gamma
		self.tau = tau
		self.memory_capacity = memory_capacity
		self.batch_size = batch_size
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr

		# initialize actor and critic networks, as well as their optimizers
		self.actor = Actor(self.nb_states, self.nb_actions, self.actor_lr).to(device)
		self.actor_target = Actor(self.nb_states, self.nb_actions, self.actor_lr).to(device)
		# load the actor's parameter dictionary to critic_target
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.actor_lr)

		self.critic = Critic(self.nb_states, self.nb_actions, self.critic_lr).to(device)
		self.critic_target = Critic(self.nb_states, self.nb_actions, self.critic_lr).to(device)
		# load the critic's parameter dictionary to critic_target
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.cricit_optimizer = optim.Adam(self.critic.parameters(), lr = self.critic_lr)

		# initialize experience replay memory
		self.memory = ReplayMemory(self.memory_capacity, input_dims, nb_actions)

		# initialize random noisifying process
		self.exploration_noise = OU_Noise(self.env.action_space, mu=np.zeros(self.nb_actions))

	def select_action(self, observation):
		# get an action from the actor and then apply noise
		# to it for exploration purposes.
		self.actor.eval()
		observation = T.tensor(observation, dtype = T.float).to(self.actor.device)
		action = self.actor.forward(observation).to(self.actor.device)
		noisified_action = action + T.tensor(self.exploration_noise(),
                                 dtype=T.float).to(self.actor.device)
		self.actor.train()
		return noisified_action.cpu().detach().numpy()

	def remember(self, s_t, a_t, s_t_, r_t, done):
		self.memory.push_transition(s_t, a_t, s_t_, r_t, done)

	def learn(self):
		if len(self.memory) < self.batch_size:
			return
	
		# sample a batch of transitions from experience memory
		s_t, a_t, s_t_, r_t, done = self.memory.sample(self.batch_size)

		s_t = T.tensor(s_t, dtype=T.float).to(self.critic.device)
		a_t = T.tensor(a_t, dtype=T.float).to(self.critic.device)
		s_t_ = T.tensor(s_t_, dtype=T.float).to(self.critic.device)
		r_t = T.tensor(r_t, dtype=T.float).to(self.critic.device)
		done = T.tensor(done).to(self.critic.device)

		self.actor_target.eval()
		self.critic_target.eval()
		self.critic.eval()

		# depending on the new state, select a new action 
		# according to current policy using the actor network
		# then compute its q value using the critic network
		a_t_ = self.actor_target.forward(s_t_)
		q_value_ = self.critic_target.forward(s_t_, a_t_)

		# also compute the q value for the current state
		q_value = self.critic.forward(s_t, a_t)

		# Set y_i = r_i + γ * q_value_. The new q value was computed
		# by passing the new actions chosen by the actor network
		# through the critic network
		y = []
		for j in range(self.batch_size):
			y.append(r_t[j] + self.gamma * q_value_[j] * done[j])
		y = T.tensor(y).to(self.critic.device)
		y = y.view(self.batch_size, 1)

		# Update critic by minimizing the mse loss
		# between y and the q value computed by the critic
		self.critic.train()
		self.critic.optimizer.zero_grad()
		critic_loss = F.mse_loss(y, q_value)
		critic_loss.backward()
		self.critic.optimizer.step()
		self.critic.eval()

		# Update actor by using the sampled policy gradient
		self.actor.optimizer.zero_grad()
		# First, compute the action for the current state
		# according to the policy modeled by the actor network
		mu = self.actor.forward(s_t)
		self.actor.train()
		# Compute the gradient of the actor loss by passing the
		# states and policy-picked corresponding actions to the 
		# critic network and averaging across this batch
		actor_loss = -self.critic.forward(s_t, mu)
		actor_loss = T.mean(actor_loss)
		actor_loss.backward()
		self.actor.optimizer.step()

		self.update_network_parameters()

	def update_network_parameters(self):

		soft_update(self.actor_target, self.actor, self.tau)
		soft_update(self.critic_target, self.critic, self.tau)
		
		"""
		actor_params = self.actor.named_parameters()
		critic_params = self.critic.named_parameters()
		actor_target_params = self.actor_target.named_parameters()
		critic_target_params = self.critic_target.named_parameters()

		actor_state_dict = dict(actor_params)
		critic_state_dict = dict(critic_params)
		actor_target_state_dict = dict(actor_target_params)
		critic_target_state_dict = dict(critic_target_params)

		#print(actor_state_dict.size())
		#print(actor_target_state_dict.size())
		print(actor_target_state_dict)
		for name in actor_target_state_dict:
			actor_target_state_dict = self.tau * actor_state_dict[name] + (1 - self.tau) * actor_target_state_dict[name]

		self.target_actor.load_state_dict(actor_state_dict)

		for name in critic_target_state_dict:
			critic_target_state_dict = self.tau * critic_state_dict[name] + (1 - self.tau) * critic_target_state_dict[name]

		self.target_critic.load_state_dict(critic_state_dict)
		"""



