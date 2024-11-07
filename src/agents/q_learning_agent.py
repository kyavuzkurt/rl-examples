import numpy as np
import math
import gymnasium as gym

class QLearningAgent:
    def __init__(self, state_size, action_size, num_bins=10, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.num_bins = num_bins
        self.q_table = np.zeros([num_bins] * state_size + [action_size])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        env = gym.make("CartPole-v1")
        #since cartpole is a continuous space andd q learning is a discrete space, we need to discretize the state space so this q learning agent is specialized for cartpole
        self.state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        self.state_bounds[1] = (-3.0, 3.0)
        self.state_bounds[3] = (-math.radians(50), math.radians(50))

    def discretize_state(self, state):
        discretized = []
        for i in range(self.state_size):
            low, high = self.state_bounds[i]
            bins = np.linspace(low, high, self.num_bins)
            discretized_state = int(np.digitize(state[i], bins)) - 1
            discretized_state = max(0, min(self.num_bins - 1, discretized_state))
            discretized.append(discretized_state)
        discretized_tuple = tuple(discretized)
        #debugging
        #print(f"State: {state}, Discretized: {discretized_tuple}")
        return discretized_tuple

    def act(self, state):
        discrete_state = self.discretize_state(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[discrete_state])

    def learn(self, state, action, reward, next_state, terminated, truncated):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        current_q = self.q_table[discrete_state][action]
        max_future_q = np.max(self.q_table[discrete_next_state])
        
        # Compute the target
        target = reward + self.discount_factor * max_future_q * (not (terminated or truncated))
        
        # Update Q-value
        self.q_table[discrete_state][action] += self.learning_rate * (target - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Debugging line
        #print(f"Update Q[{discrete_state}][{action}]: {current_q} -> {self.q_table[discrete_state][action]}")