import gymnasium as gym
import numpy as np
from agents.q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt
import time
import pickle

# Initialize environment
env = gym.make("CartPole-v1", render_mode="rgb_array")  # Change render_mode to "human" for visualization
agent = QLearningAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, num_bins=20)

rewards = []
epsilon_values = []

with open('q_table.pkl', 'rb') as f:
    print("Loading Q-table from file")
    agent.q_table = pickle.load(f)

for episode in range(5000):  
    state, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.learn(state, action, reward, next_state, terminated, truncated)
        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    epsilon_values.append(agent.epsilon)
    print(f"Episode {episode+1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.4f}")



# Save Q-table
with open('q_table.pkl', 'wb') as f:
    print("Saving Q-table to file")
    pickle.dump(agent.q_table, f)

env.close()

# Plotting
window = 50
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

smoothed_rewards = moving_average(rewards)
plt.plot(smoothed_rewards)
plt.xlabel("Episode")
plt.ylabel("Smoothed Total Reward")
plt.title("Smoothed Training Progress")
plt.show()

"""
# Plotting Epsilon Decay
plt.plot(epsilon_values, label='Epsilon Value')
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay Over Time")
plt.legend()
plt.show()
"""