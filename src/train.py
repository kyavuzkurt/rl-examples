import gymnasium as gym
import numpy as np
from agents.q_learning_agent import QLearningAgent
from enviroments.plot import Plotter
import pickle
import os

def train(episodes=5000, load_qtable=True, save_qtable=True):
    # Initialize environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent = QLearningAgent(state_size=env.observation_space.shape[0], 
                          action_size=env.action_space.n, 
                          num_bins=20)
    
    if load_qtable and os.path.exists('q_table.pkl'):
        with open('q_table.pkl', 'rb') as f:
            print("Loading Q-table from file")
            agent.q_table = pickle.load(f)
    
    rewards = []
    epsilon_values = []
    
    for episode in range(episodes):
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
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode+1}: Average Reward (last 100) = {avg_reward:.2f}, Epsilon = {agent.epsilon:.4f}")
    
    # Save Q-table if requested
    if save_qtable:
        with open('q_table.pkl', 'wb') as f:
            print("Saving Q-table to file")
            pickle.dump(agent.q_table, f)
    
    env.close()
    return rewards, epsilon_values

if __name__ == "__main__":
    rewards, epsilon_values = train()
    plotter = Plotter()
    plotter.plot_training_results(rewards, epsilon_values)