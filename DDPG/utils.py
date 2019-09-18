import random
import numpy as np
import numpy.random as nr
from torch.autograd import Variable

Transition = namedtuple('Transition',
						('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.memory_capacity = self.capacity
		self.current_position = 0
		self.experience_memory = []

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.current_position] = Transition(*args)
		self.position = (self.position + 1) % self.memory_capacity

	def sample(self, batch_size):
		return random.sample(self.experience_memory, batch_size)

	def __len__(self):
		return len(self.memory)

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

	def __init__(self, action_space, mu = 0, theta = 0.15, min_sigma = 0.2, max_sigma = 0.2, decay_period = 100000):
		self.action_dimension = action_space.shape[0]
		self.mu = mu
		self.theta = theta
		self.max_sigma = max_sigma
		self.min_sigma = min_sigma
		self.state = np.ones(self.action_dimension) * self.mu
		self.low = action_space.low
		self.high = action_space.high
		self.t = 0
		self.reset()

	def reset(self):
		self.state = np.ones(self.action_dimension) * self.mu

	def noisify_state(self, state):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
		self.state = x + dx
		return self.state

	def get_action(self, state):
		ou_state = self.noisify_state(state)
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
		self.t += 1
		return np.clip(action + ou_state, self.low, self.high)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)