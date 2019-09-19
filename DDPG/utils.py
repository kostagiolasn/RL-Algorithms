import random
import numpy as np
import numpy.random as nr
from torch.autograd import Variable
import matplotlib.pyplot as plt

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class ReplayMemory(object):
	def __init__(self, capacity, input_shape, nb_actions):
		self.capacity = capacity
		self.memory_offset = 0
		self.state_memory = np.zeros((self.capacity, *input_shape))
		self.new_state_memory = np.zeros((self.capacity, *input_shape))
		self.action_memory = np.zeros((self.capacity, nb_actions))
		self.reward_memory = np.zeros(self.capacity)
		self.done_memory = np.zeros(self.capacity, dtype = np.float32)

	def push_transition(self, s_t, a_t, s_t_, r_t, done):
		self.state_memory[memory_offset] = s_t
		self.new_state_memory[memory_offset] = s_t_
		self.action_memory[memory_offset] = a_t
		self.reward_memory[memory_offset] = r_t
		self.done_memory[memory_offset] = 1 - done
		self.memory_offset += 1

	def sample(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        state_batch = self.state_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        done_batch = self.done_memory[batch]

        return state_batch, action_batch, new_state_batch, reward_batch, done_batch

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

	def __init__(self, action_space, mu, theta = 0.15, dt = 1e-2, x0 = None):
		self.action_dimension = action_space.shape[0]
		self.mu = mu
		self.theta = theta
		self.dt = dt

		self.reset()

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt
		dx = self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
		noise = x + dx
		self.x_prev = x
		return noise