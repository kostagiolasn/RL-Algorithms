import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Box
from OU_noise import *
from agent import DDPGagent
from utils import plotLearning

def main():
	# Specify environment name here
	ENV_NAME = 'Walker2d-v2'
	TOTAL_EPISODES = 10000

	env = gym.make(ENV_NAME)

	agent = DDPGagent(env)

	score_history = []

	np.random.seed(0)
	for i in range(TOTAL_EPISODES):
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			new_state, reward, done, info = env.step(action)
			agent.remember(observation, action, reward, new_state, int(done))
			agent.learn()
			score += reward
			observation = new_state

		score_history.append(score)

		print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

	filename = 'Walker2d-v2.png'
	plotLearning(score_history, filename, window = 100)

if __name__ == '__main__':
	main()
