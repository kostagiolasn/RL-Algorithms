import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Box
from OU_noise import *

def main():
	# Specify environment name here
	ENV_NAME = 'Walker2d-v2'
	TOTAL_EPISODES = 10000

	env = gym.make(ENV_NAME)

	# Ensure that observation space is continuous
	#assert isinstance(env.observation_space, Box), "observation space must be continuous"
	#assert isinstance(env.action_space, Box), "action space must be continuous"

	env.reset()
	print(env.action_space.shape)
	#print("Total number of possible states %d" % env.observation_space.n)

	#print("Total number of possible actions %d" % env.action_space.n)

	OU_noise = OU_Noise(env.action_space.shape)

if __name__ == '__main__':
	main()
