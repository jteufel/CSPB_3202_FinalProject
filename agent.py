'''
Agent from step 1 of Report.
Credit github user gregd190 for initial code
'''

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

"""
Function to generate action bins for use in training episodes
"""
def create_action_bins(z):
    actionbins = np.linspace(-2.0, 2.0, z)
    return actionbins

"""
Function to generate action index for use in training episodes
"""
def find_actionbin(action, actionbins):
    idx = (np.abs(actionbins - action)).argmin()
    return idx
"""
Function to build the Sequential nueral network model
No major adjustments from initial agent
"""
def build_model(num_output_nodes):

    model = Sequential()
    model.add(Dense(128, input_shape = (3,), activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(num_output_nodes, activation = 'linear'))
    adam = optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999)
    model.compile(loss = 'mse', optimizer = adam)

    return model

"""
Function to implement a training iteration of the DNN model.

Modifications from original agent include reformatting the code such that for each training iteration,
the model would only be fit once on multidimensional data. The existing training loop was refitting the
model on each new sample collected.

Updated model fitting include 20 epochs.
"""

def train_model(data, model):

    state, action, reward, next = data['state'], data['action'], data['reward'], data['next']
    target = reward + np.amax(model.predict(next), axis=1)
    targetfull = model.predict(state)

    for i in range(0,len(action)): targetfull[i][int(action[i])] = target[i]

    model.fit(state, targetfull, epochs = 20, verbose = 0)

    return model

"""
Function to generate data and and implement training iterations.

Number of training iterations per fucntion call defined by the iters var. Sample # passed into each training
session defined by trainingSamples var.

As eps decay var decreases, sample actions are determined by the model more often by the model then randomly generated.

Modifications from original agent include reformatting the data passed into the training iteration,
and incremental tweeks to the samples passed into each training session (trainingSamples var on line 94)
"""

def run_episodes(env, model, eps = 0.999, r = False, iters = 100):

    eps_decay = 0.9999
    eps_min = 0.02

    for i in range(iters):
        state = env.reset()

        stateArray, actionArray, rewardArray, nextStateArray = (np.array([]) for i in range(4))

        if eps>eps_min:
            eps = eps * eps_decay

        trainingSamples = 2000
        for _ in range(trainingSamples):

            if np.random.uniform() < eps:
                action = env.action_space.sample()
            else:
                flat_state = np.reshape(state, [1,3])
                action = np.amax(model.predict(flat_state))

            actionbin = find_actionbin(action, actionbinslist)
            action = actionbinslist[actionbin]
            action = np.array([action])
            observation, reward, done, _ = env.step(action)
            state_new = observation

            stateArray = np.append(stateArray,state)
            actionArray = np.append(actionArray,actionbin)
            rewardArray = np.append(rewardArray,reward)
            nextStateArray = np.append(nextStateArray,state_new)

            state = state_new

        stateArray,nextStateArray = np.reshape(stateArray, [trainingSamples,3]),np.reshape(nextStateArray, [trainingSamples,3])


        data = {'state':stateArray, 'action':actionArray, 'reward':rewardArray, 'next':nextStateArray}

        model = train_model(data, model)

    return eps, model

"""
Main function generate the environment initiate training.
No major adjustments from initial agent.
Reduced the number of training iterations per run to 500,
and increased interval to 100 - run_episodes is only called 5 times.
"""

if __name__ == '__main__':

	env = gym.make('Pendulum-v0')

	eps = 1
	num_action_bins = 10
	actionbinslist = create_action_bins(num_action_bins)

	model = build_model(num_action_bins)
	totaliters = 500
	test_interval = 100
	numeps = int(totaliters)
	print('numeps = ', numeps)
	cnt = 0
	while cnt < totaliters:
		eps, model = run_episodes(env, model, eps = eps, r = False, iters = test_interval)
		cnt += test_interval
		print(cnt, 'iterations')

	print('saving model')
	model.save('pendulum-model-test1.h5')
	print('model saved')
