import gym, keras, numpy as np
import matplotlib.pyplot as plt

from gym_wrapper import *

frames_used = 4
max_replay_size = 500

def model_init():
    std_dev = 0.001
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (8,8), strides=(4, 4), padding = "same", activation='elu', input_shape=(84, 84, frames_used), 
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer='zeros'))
    model.add(keras.layers.Conv2D(32, (3,3), strides=(3, 3), padding = "same", activation='elu', 
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer='zeros'))
    model.add(keras.layers.Conv2D(64, (3,3), strides=(1, 1), padding = "same", activation='elu', 
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer='zeros'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=256, activation='elu', 
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer='zeros'))
    model.add(keras.layers.Dense(units=3, activation='elu', 
                                  kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=std_dev, seed=None), bias_initializer='zeros'))
    print(model.summary())
    return model

env = MainGymWrapper.wrap(gym.make('BreakoutDeterministic-v4'))
env.reset()

replays = []
def train(model, random_action_chance, repetitions=100, learning_rate=0.01, number_of_repetitions_after_death = 50, reward_decay = 0.95):
    print('Training with repetetitions = ' + str(repetitions) + ', learning_rate = ' + str(learning_rate) + ', random_action_chance = ' + str(random_action_chance))
    model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=learning_rate))
    sum_scores = 0
    for i in range(repetitions):
        done = False
        observation = env.reset()
        observation = np.moveaxis(observation, 0, 2)
        observation = np.reshape(observation, (1,84,84,4))
        replay = [observation]
        prediction = model.predict(replay[0])
        a = np.argmax(prediction)+1
        score = 0
        j = 0
        predictions = []
        while not done:
            j += 1
            env.render()
            if j == 200:
                a = 1
                print("Forced Start")
            next_observation, reward, done, info = env.step(a)
            if reward > 0 or a == 1:
                score += reward
                j = 0
            next_observation = np.moveaxis(next_observation, 0, 2)
            next_observation = np.reshape(next_observation, (1,84,84,4))
            replay.append(next_observation)
            old_prediction = prediction
            prediction = model.predict(replay[-1])
            print('\r', prediction, sep='', end='', flush=True)
            old_a = a
            if np.random.rand() < random_action_chance:
                a = np.random.randint(1, 4)
            else:
                a = np.argmax(prediction)+1
            if done:
                old_prediction = np.zeros([1,3])
                epochs = number_of_repetitions_after_death
                model.save('breakout.h5')
            else:
                old_prediction[0][old_a-1] = reward + reward_decay * np.max(prediction)
                epochs = 1
            predictions.append(old_prediction)
            model.fit(replay[-2], old_prediction, epochs = epochs, verbose = False)
        remember(predictions, replay)
        print()
        sum_scores += score
        print(i, score, sum_scores/(i+1))

def remember(prediction, replay):
    if len(replays) < max_replay_size:
        replays.append((prediction, replay))
    else:
        replays.pops(0)
        replays.append((prediction, replay))

model = model_init()
train(model, 1, repetitions=20, learning_rate=0.1, number_of_repetitions_after_death = 0)
train(model, 1, repetitions=20, learning_rate=0.01, number_of_repetitions_after_death = 0)
train(model, 1, repetitions=20, learning_rate=0.001, number_of_repetitions_after_death = 0)
train(model, 0.9, repetitions = 100, learning_rate=0.001, number_of_repetitions_after_death = 0)
train(model, 0.9, repetitions = 100, learning_rate=0.0005, number_of_repetitions_after_death = 0)
train(model, 0.8, repetitions = 100, learning_rate=0.00025, number_of_repetitions_after_death = 0)
train(model, 0.7, repetitions = 100, learning_rate=0.0001, number_of_repetitions_after_death = 0)
train(model, 0.6, repetitions = 100, learning_rate=0.0001, number_of_repetitions_after_death = 0)
train(model, 0.5, repetitions = 100, learning_rate=0.0001, number_of_repetitions_after_death = 0)

env.close()

model.save('breakout.h5')