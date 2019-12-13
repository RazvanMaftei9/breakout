import gym, keras, numpy as np

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(4, (3,3), strides=(1, 1), padding = "same", activation='elu', input_shape=(98, 72, 2)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(4, (3,3), strides=(1, 1), padding = "same", activation='elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(4, 4)))
model.add(keras.layers.Conv2D(8, (3,3), strides=(1, 1), padding = "same", activation='elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(4, 4)))
model.add(keras.layers.Conv2D(16, (3,1), strides=(1, 1), activation='elu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=16, activation='elu'))
model.add(keras.layers.Dense(units=3, activation='elu'))


#model = keras.models.Sequential()
#model.add(keras.layers.Dense(units=32, activation='elu', input_dim=128))
#model.add(keras.layers.Dense(units=8, activation='elu'))
#model.add(keras.layers.Dense(units=3, activation='elu'))

#model = keras.models.load_model("breakout.h5")

def image_prep(obs):
    obs = obs[0:-14,8:-8]
    obs = obs[0::2,0::2]
    obs = np.expand_dims(obs, axis=0)
    obs = np.mean(obs, axis=3, keepdims = True)
    obs[obs > 0] = 1
    return obs

model.compile(loss='mse',
              optimizer=keras.optimizers.SGD(lr=0.1))
print(model.summary())

env = gym.make('Breakout-v0')
env.reset()
for i in range(10):
    done = False
    observation = image_prep(env.reset())
    last_observation = observation
    prediction = model.predict(np.concatenate((observation,last_observation),axis=3))
    a = np.argmax(prediction)+1
    print(prediction)
    sum = 0
    j = 0
    while not done:
        j += 1
        env.render()
        if j == 200:
            a = 1
            print("Forced Start")
        next_observation, reward, done, info = env.step(a)
        if reward >  0 or a == 1:
            sum += reward
            j = 0
        next_observation = image_prep(next_observation)
        prediction = model.predict(np.concatenate((next_observation,observation),axis=3))
        print(prediction)
        old_a = a
        a = np.argmax(prediction)+1
        a = np.random.randint(1, 4)
        if done:
            prediction = np.zeros([1,3])
            reps = 50
        else:
            prediction[0][old_a-1] = reward + 0.99 * np.max(prediction)
            reps = 1
        model.fit(np.concatenate((observation,last_observation),axis=3), prediction, epochs = reps, verbose = False)
        last_observation = observation
        observation = next_observation
    print(i, sum)
    if (i+1) % 100 == 0:
        model.save('breakout.h5')
env.close()

model.save('breakout.h5')