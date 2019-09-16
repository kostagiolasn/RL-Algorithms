import numpy as np
import numpy.random as nr

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
		self.reset()

	def reset(self):
		self.state = np.ones(self.action_dimension) * self.mu

	def noisify_state(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
		self.state = x + dx
		return self.state

	def get_action(self, action, t = 0):
		ou_state = self.noisify_state()
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
		return np.clip(action + ou_state, self.low, self.high)
