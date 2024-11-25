# Reinforcement Learning Examples

## Overview

This repository showcases my learning journey through various reinforcement learning (RL) algorithms. Starting with the implementation of the Q-Learning algorithm applied to the CartPole environment using Gymnasium, the project aims to progressively incorporate more complex examples and advanced RL techniques.

## Current Implementation

### Q-Learning on CartPole

The current focus of this repository is on the CartPole-v1 environment provided by Gymnasium. The implementation leverages a Q-Learning agent to solve the environment, demonstrating fundamental RL concepts such as state discretization, action selection, and policy improvement.

**Key Components:**

- **Environment Setup:** Utilizes Gymnasium to create and manage the CartPole environment.
- **QLearningAgent:** A custom agent that implements the Q-Learning algorithm, including state discretization and epsilon-greedy action selection.
- **Training Script (`train.py`):** Manages the training loop, agent interactions with the environment, and performance tracking.
- **Persistence:** Saves and loads the Q-table using Python's `pickle` module to maintain learned policies across sessions.
- **Visualization:** Employs Matplotlib to plot training progress and epsilon decay over episodes.

### DQN on CartPole

This project implements a DQN agent on the CartPole environment. The agent uses a deep neural network to approximate the Q-values and learns to make optimal decisions based on the environment's dynamics.

**Key Components:**

- **Environment Setup:** Utilizes Gymnasium to create and manage the CartPole environment.
- **DQNAgent:** A custom agent that implements the DQN algorithm, including experience replay, target network updates, and epsilon-greedy action selection.
- **Training Script (`train.py`):** Manages the training loop, agent interactions with the environment, and performance tracking.
- **Persistence:** Saves and loads the model using TensorFlow's `SavedModel` format to maintain learned policies across sessions.
- **Visualization:** Employs Matplotlib to plot training progress and epsilon decay over episodes.

## Future Plans

- **Expand RL Algorithms:** Incorporate additional algorithms such as SARSA, Deep Q-Networks (DQN), and Policy Gradient methods.
- **Diverse Environments:** Explore other OpenAI Gym environments to apply and test different RL strategies.
- **Advanced State Representations:** Implement function approximation techniques to handle continuous and high-dimensional state spaces.
- **Performance Enhancements:** Optimize training processes and explore parallelization to accelerate learning.
- **Comprehensive Documentation:** Provide detailed explanations, tutorials, and usage examples for each implemented algorithm.



## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for providing the gymnasium environments.
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/) for the foundational concepts and RL algorithms.
