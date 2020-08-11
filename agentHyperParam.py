import h5py
import numpy as np
from collections import deque
from keras.models import Sequential
import tensorflow
import gym
import gym.spaces
import gym.wrappers
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from collections import deque
from keras.layers import Flatten, Dense
from keras import optimizers, losses
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers
import kerastuner as kt
from kerastuner.tuners import Hyperband
from Hyper import ClearTrainingOutput



def create_action_bins(z):
    actionbins = np.linspace(-2.0, 2.0, z)
    return actionbins

def find_actionbin(action, actionbins):
    idx = (np.abs(actionbins - action)).argmin()
    return idx

def build_model(hp):

    model = Sequential()

    model.add(Flatten())

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value = 64, max_value = 256, step = 32)

    model.add(Dense(units = hp_units, activation = 'relu'))
    model.add(Dense(10))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    model.compile(optimizer = optimizers.Adam(learning_rate = hp_learning_rate),loss = 'mse', metrics = ['accuracy'])

    return model

def train_model(data, tuner, model = None):

    state, action, reward, next = data['state'], data['action'], data['reward'], data['next']

    if model != None:
        MaxTarget = reward + np.amax(model.predict(next), axis=1)
        MaxTargets = model.predict(state)
        for i in range(0,len(action)): MaxTargets[i][int(action[i])] = MaxTarget[i]

    else:
        MaxTarget = reward + np.amax(next, axis=1)
        MaxTargets = np.full([500,10],float('-inf'))
        for i in range(0,len(action)): MaxTargets[i][int(action[i])] = MaxTarget[i]



    for i in range(0,len(action)): MaxTargets[i][int(action[i])] = MaxTarget[i]

    tuner.search(state, MaxTargets, epochs = 10, callbacks = [ClearTrainingOutput()])
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    model = tuner.hypermodel.build(best_hps)
    model.fit(state, MaxTargets, epochs = 10)

    return tuner.get_best_models(num_models=1)[0]

def run_episodes(tuner, model = None, eps = 0.999, r = False, iters = 100):

    eps_decay = 0.9999
    eps_min = 0.02
    for i in range(iters):
        state = env.reset()

        totalreward = 0
        stateArray, actionArray, rewardArray, nextStateArray = (np.array([]) for i in range(4))

        if eps>eps_min:
            eps = eps * eps_decay

        iterations = 500
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

        model = train_model(data, tuner, model)

    return eps, model


if __name__ == '__main__':

    env = gym.make('Pendulum-v0')

    eps = 1
    num_action_bins = 10
    actionbinslist = create_action_bins(num_action_bins)


    tuner = kt.Hyperband(build_model,
                     objective = 'accuracy',
                     max_epochs = 20,
                     overwrite = True)

    totarray = []
    cntarray = []
    totaliters = 250
    test_interval = 25
    numeps = int(totaliters)
    print('numeps = ', numeps)
    cnt = 0
    model = None

    while cnt < totaliters:
        eps, model = run_episodes(tuner, model, eps = eps, r = False, iters = test_interval)
        cnt += test_interval
        trarray = []



    print('saving model')
    model.save('pendulum-model-test.h5')
    print('model saved')
