import random

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