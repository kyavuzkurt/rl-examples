import gymnasium as gym
import numpy as np
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from enviroments.plot import Plotter
import pickle
import os

class Trainer:
    def __init__(self, agent_type='q_learning', **agent_params):
        self.agent_type = agent_type
        if agent_type == 'q_learning':
            self.agent = QLearningAgent(**agent_params)
        elif agent_type == 'dqn':
            self.agent = DQNAgent(**agent_params)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def train(self, episodes=1000, load_model=True, save_model=True):
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        
        if load_model and self.agent_type == 'q_learning' and os.path.exists('q_table.pkl'):
            with open('q_table.pkl', 'rb') as f:
                print("Loading Q-table from file")
                self.agent.q_table = pickle.load(f)
        elif load_model and self.agent_type == 'dqn' and os.path.exists('dqn_model.h5'):
            print("Loading DQN model from file")
            self.agent.model.load_weights('dqn_model.h5')
        
        rewards = []
        epsilon_values = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.agent.learn(state, action, reward, next_state, terminated, truncated)
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
            epsilon_values.append(self.agent.epsilon)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode+1}: Average Reward (last 100) = {avg_reward:.2f}, Epsilon = {self.agent.epsilon:.4f}")
        
        if save_model:
            if self.agent_type == 'q_learning':
                with open('q_table.pkl', 'wb') as f:
                    print("Saving Q-table to file")
                    pickle.dump(self.agent.q_table, f)
            else:
                print("Saving DQN model to file")
                self.agent.model.save_weights('dqn_model.h5')
        
        env.close()
        return rewards, epsilon_values

if __name__ == "__main__":
    # Example usage for Q-Learning
    #trainer_q = Trainer(agent_type='q_learning', state_size=4, action_size=2, num_bins=20)
    #rewards_q, epsilon_values_q = trainer_q.train()
    
    # Example usage for DQN
    trainer_dqn = Trainer(agent_type='dqn', state_size=4, action_size=2)
    rewards_dqn, epsilon_values_dqn = trainer_dqn.train()
    
    plotter = Plotter()
    #plotter.plot_training_results(rewards_q, epsilon_values_q, save_path='q_learning_results.png')
    plotter.plot_training_results(rewards_dqn, epsilon_values_dqn, save_path='dqn_results.png')
