import numpy as np
import gymnasium as gym
from gymnasium import wrappers

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', render_mode="human")

# Initialize the Q-table with zeros
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Q-learning parameters
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < 0.3: # Explore with 30% probability
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Take the chosen action
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # Update the Q-table using the Q-learning formula
        Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]))

        state = next_state 

# Now, let's see how well our agent performs
total_rewards = 0
num_test_episodes = 100

for episode in range(num_test_episodes):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state, :])
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        state = next_state

average_reward = total_rewards / num_test_episodes
print(f"Average reward over {num_test_episodes} test episodes: {average_reward}")