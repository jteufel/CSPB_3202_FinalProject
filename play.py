#code directly from

import gym
import gym.spaces
import gym.wrappers
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from collections import deque
from keras.layers import Flatten, Dense
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers

def play_game(eps = 0.0,  r = True):

	state = env.reset()
	totalreward = 0
	cnt = 0
	while cnt < 200:
		cnt += 1
		if r:
			env.render()
		if np.random.uniform() < eps:
			action = env.action_space.sample()
			actionbin = find_actionbin(action, actionbinslist)
		else:
			flat_state = np.reshape(state, [1,3])
			actionbin = np.argmax(model.predict(flat_state))
		action = actionbinslist[actionbin]
		action = np.array([action])
		observation, reward, done, _ = env.step(action)
		totalreward += reward
		state_new = observation
		state = state_new

	return totalreward


def create_action_bins(z):
    actionbins = np.linspace(-2.0, 2.0, z)
    return actionbins

env = gym.make('Pendulum-v0')
eps = 1
num_action_bins = 10
actionbinslist = create_action_bins(num_action_bins)
print('loading model')
model = load_model('pendulum-model-test2.h5')
print('model loaded')

for _ in range(10):
    play_game(r = True)
