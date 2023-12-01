import gymnasium as gym
import numpy as np
import random

envTrain = gym.make("CarRacing-v2", continuous=False, lap_complete_percent=0.2, render_mode="human") # full_action_space=True render_mode="human"

q_table = np.zeros((envTrain.observation_space.high.size, envTrain.action_space.n))

alpha = 0.1 # learning rate
gamma = 0.9 # discount
epsilon = 0.35 # probability for exploration

all_epochs = []
all_penalities = []
all_rewards = []

for i in range(1, 5001):
    observation, info = envTrain.reset(options={"randomize":False})
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = envTrain.action_space.sample()
        else:
            action = np.argmax(q_table[observation])
        
        next_observation, reward, terminated, truncated, info = envTrain.step(action)
        done = terminated or truncated

        # print(action)
        old_value = q_table[observation][action]
        next_max = np.max(q_table[next_observation])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[observation][action] = new_value

        if reward <= -5:
            penalties += 1
        observation = next_observation
        epochs += 1

    all_epochs.append(epochs)
    all_penalities.append(penalties)
    all_rewards.append(reward)
    if i % 200 == 0:
        print(f"Episode: {i} with reward of {reward}")

print("Training finished")
envTrain.close()

envTest = gym.make("CarRacing-v2", render_mode="human")

for i in range(100):
    observation, info, = envTest.reset()

    done = False
    while not done:
        action = np.argmax(q_table[observation])

        next_observation, reward, terminated, truncated, info = envTest.step(action)

        done = terminated or truncated

envTest.close()