import random
import numpy as np
import numpy.random as nr
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gym

def visualize_agent(env, agent, timesteps = 10000):
	observation = env.reset()
	for t in range(timesteps):
		env.render()
		action = agent.select_action(observation)
		new_state, reward, done, info = env.step(action)

	env.close()

def plotLearning(x, scores, filename):
	window = 10
	smoothed_scores = [np.mean(scores[i-window:i+1]) if i > window 
                    else np.mean(scores[:i+1]) for i in range(len(scores))]

	plt.figure(figsize=(12,8))
	plt.plot(x, scores)
	plt.plot(x, smoothed_scores)
	plt.ylabel('Total Scores')
	plt.xlabel('Episodes')
	plt.show()
	plt.savefig(filename)

class ReplayMemory(object):
	def __init__(self, capacity, input_shape, nb_actions):
		self.capacity = int(capacity)
		self.memory_offset = 0
		self.state_memory = np.zeros((self.capacity, *input_shape))
		self.new_state_memory = np.zeros((self.capacity, *input_shape))
		self.action_memory = np.zeros((self.capacity, nb_actions))
		self.reward_memory = np.zeros(self.capacity)
		self.done_memory = np.zeros(self.capacity, dtype = np.float32)

	def push_transition(self, s_t, a_t, s_t_, r_t, done):

		rotating_offset = self.memory_offset % self.capacity

		self.state_memory[rotating_offset] = s_t
		self.new_state_memory[rotating_offset] = s_t_
		self.action_memory[rotating_offset] = a_t
		self.reward_memory[rotating_offset] = r_t
		self.done_memory[rotating_offset] = 1 - done
		self.memory_offset += 1

	def sample(self, batch_size):
		max_mem = min(self.memory_offset, self.capacity)

		batch = np.random.choice(max_mem, batch_size)

		state_batch = self.state_memory[batch]
		new_state_batch = self.new_state_memory[batch]
		action_batch = self.action_memory[batch]
		reward_batch = self.reward_memory[batch]
		done_batch = self.done_memory[batch]

		return state_batch, action_batch, new_state_batch, reward_batch, done_batch

	def __len__(self):
		return self.memory_offset

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)

class OU_Noise:

	"""
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

	def __init__(self, action_space, mu, sigma = 0.15, theta = 0.2, dt = 1e-2, x0 = None):
		self.action_dimension = action_space.shape[0]
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0

		self.reset()

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt
		dx = self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
		noise = x + dx
		self.x_prev = x
		return noise

def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)