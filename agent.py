import h5py
import numpy as np
from collections import deque
from keras.models import Sequential
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
import kerastuner as kt
from kerastuner.tuners import Hyperband


def create_action_bins(z):
    actionbins = np.linspace(-2.0, 2.0, z)
    return actionbins

def find_actionbin(action, actionbins):
    idx = (np.abs(actionbins - action)).argmin()
    return idx

def build_model(num_output_nodes):

    model = Sequential()
    model.add(Dense(128, input_shape = (3,), activation = 'relu'))

    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(num_output_nodes, activation = 'linear'))
    adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999)
    model.compile(loss = 'mse', optimizer = adam)

    return model

def train_model(data, model):

    state, action, reward, next = data['state'], data['action'], data['reward'], data['next']
    target = reward + np.amax(model.predict(next), axis=1)
    targetfull = model.predict(state)

    for i in range(0,len(action)): targetfull[i][int(action[i])] = target[i]

    model.fit(state, targetfull, epochs = 20, verbose = 0)

    return model

def run_episodes(env, model, eps = 0.999, r = False, iters = 100):

    eps_decay = 0.9999
    eps_min = 0.02

    for i in range(iters):
        state = env.reset()

        totalreward = 0
        stateArray, actionArray, rewardArray, nextStateArray = (np.array([]) for i in range(4))

        if eps>eps_min:
            eps = eps * eps_decay

        iterations = 2000
        for _ in range(iterations):

            if np.random.uniform() < eps:
                action = env.action_space.sample()
            else:
                flat_state = np.reshape(state, [1,3])
                action = np.amax(model.predict(flat_state))

            actionbin = find_actionbin(action, actionbinslist)
            action = actionbinslist[actionbin]
            action = np.array([action])
            observation, reward, done, _ = env.step(action)
            totalreward += reward
            state_new = observation

            stateArray = np.append(stateArray,state)
            actionArray = np.append(actionArray,actionbin)
            rewardArray = np.append(rewardArray,reward)
            nextStateArray = np.append(nextStateArray,state_new)

            state = state_new

        stateArray,nextStateArray = np.reshape(stateArray, [iterations,3]),np.reshape(nextStateArray, [iterations,3])


        data = {'state':stateArray, 'action':actionArray, 'reward':rewardArray, 'next':nextStateArray}

        model = train_model(data, model)

    return eps, model

if __name__ == '__main__':

	env = gym.make('Pendulum-v0')

	eps = 1
	num_action_bins = 10
	actionbinslist = create_action_bins(num_action_bins)

	model = build_model(num_action_bins)
	totarray = []
	cntarray = []
	totaliters = 500
	test_interval = 100
	numeps = int(totaliters)
	print('numeps = ', numeps)
	cnt = 0
	while cnt < totaliters:
		eps, model = run_episodes(env, model, eps = eps, r = False, iters = test_interval)
		cnt += test_interval
		trarray = []

		print(cnt, 'iterations')

	print('saving model')
	model.save('pendulum-model-test2.h5')
	print('model saved')
