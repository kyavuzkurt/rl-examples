import gymnasium as gym
import numpy as np
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
import pickle
import time
from enviroments.plot import Plotter

def evaluate(episodes=10, render=True):
    # Initialize environment
    env = gym.make("CartPole-v1", render_mode="human" if render else "rgb_array")
    agent = QLearningAgent(state_size=env.observation_space.shape[0], 
                          action_size=env.action_space.n, 
                          num_bins=20)
    
    # Load trained Q-table
    try:
        with open('q_table.pkl', 'rb') as f:
            agent.q_table = pickle.load(f)
    except FileNotFoundError:
        print("No trained Q-table found. Please train the agent first.")
        return
    
    # Set epsilon to 0 for greedy action selection
    agent.epsilon = 0
    
    evaluation_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            action = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if render:
                time.sleep(0.02)  # Add delay for visualization
        
        evaluation_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    env.close()
    
    print(f"\nAverage Reward over {episodes} episodes: {np.mean(evaluation_rewards):.2f}")
    print(f"Standard Deviation: {np.std(evaluation_rewards):.2f}")
    
    # Create plotter and visualize results
    plotter = Plotter()
    plotter.plot_evaluation_results(evaluation_rewards)

if __name__ == "__main__":
    evaluate()
