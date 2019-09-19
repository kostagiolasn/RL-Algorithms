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
		# load the actor's parameter dictionary to critic_target
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_lr)

		self.critic = Critic(nb_states, nb_actions).to(device)
		self.critic_target = Critic(nb_states, nb_actions).to(device)
		# load the critic's parameter dictionary to critic_target
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.cricit_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)

		# initialize experience replay memory
		self.memory = ReplayMemory(self.memory_capacity)

		# initialize random noisifying process
		self.exploration_noise = OU_Noise(mu=np.zeros(self.nb_actions))

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

	def learn():
		if len(self.memory) < self.batch_size:
			return
	
		# sample a batch of transitions from experience memory
		s_t, a_t, s_t_, r_t, done = self.memory.sample(self.batch_size)

		s_t = T.tensor(s_t, float).to(self.critic.device)
		a_t = T.tensor(a_t, float).to(self.critic.device)
		s_t_ = T.tensor(s_t_, float).to(self.critic.device)
		r_t = T.tensor(r_t, float).to(self.critic.device)
		done = T.tensor(done).to(self.critic.device)

		self.actor_target.eval()
		self.critic_target.eval()
		self.critic.eval()

		# select an action according to current policy
		# using the actor network, then compute its q value
		# using the critic network
		a_t_ = self.actor_target.forward(s_t_)
		q_value_ = self.critic_target.forward(s_t_, a_t_)
		q_value = self.critic.forward(s_t, a_t)

		# Set y_i = r_i + γ * q_value_. The new q value is computed
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
		mu = self.actor.forward(s_t)
		self.actor.train()
		actor_loss = -self.critic.forward(s_t, mu)
		actor_loss = T.mean(actor_loss)
		actor_loss.backward()
		self.actor.optimizer.step()

		self.update_network_parameters

	def update():
		actor_params = self.actor.named_parameters()
		critic_params = self.critic.named_parameters()
		actor_target_params = self.actor_target.named_parameters()
		critic_target_params = self.critic_target.named_parameters()

		actor_state_dict = dict(actor_params)
		critic_state_dict = dict(critic_params)
		actor_target_state_dict = dict(actor_target_params)
		critic_target_state_dict = dict(critic_target_params)

		for name in actor_target_state_dict:
			actor_target_state_dict = self.tau * actor_state_dict[name] + (1 - self.tau) * actor_target_state_dict[name]

		self.target_actor.load_state_dict(actor_state_dict)

		for name in critic_target_state_dict:
			critic_target_state_dict = self.tau * critic_state_dict[name] + (1 - self.tau) * critic_target_state_dict[name]

		self.target_critic.load_state_dict(critic_state_dict)




