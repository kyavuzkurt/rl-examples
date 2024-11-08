import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, num_bins=20):
        self.state_size = state_size
        self.action_size = action_size
        self.num_bins = num_bins
        
        # Learning parameters
        self.alpha = 0.1 # Learning rate
        self.gamma = 1  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.99995
        
        # State discretization parameters
        self.state_bounds = [
            [-4.8, 4.8],     # Cart Position
            [-10, 10],       # Cart Velocity
            [-0.418, 0.418], # Pole Angle
            [-10, 10]        # Pole Angular Velocity
        ]
        
        # Initialize Q-table
        self.q_table = np.zeros([num_bins] * state_size + [action_size])

    def discretize_state(self, state):
        discrete_state = []
        for i, (lower, upper) in enumerate(self.state_bounds):
            bins = np.linspace(lower, upper, self.num_bins)
            discrete_state.append(np.maximum(np.digitize(state[i], bins) - 1, 0))
        return tuple(discrete_state)

    def act(self, state):
        discrete_state = self.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])

    def learn(self, state, action, reward, next_state, terminated, truncated):
        current_state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[current_state + (action,)]
        
        # Next Q-value
        if terminated or truncated:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state])
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[current_state + (action,)] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay