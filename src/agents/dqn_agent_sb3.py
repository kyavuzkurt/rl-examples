import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

class DQNAgentSB3:
    def __init__(self, env):
        self.env = env
        self.model = DQN(policy="MlpPolicy",
                         env=self.env,
                         learning_rate=0.0001,
                         buffer_size=1000000,
                         learning_starts=100,
                         batch_size=32,
                         tau=1.0,
                         gamma=0.99,
                         train_freq=4,
                         gradient_steps=1,
                         verbose=1)
        self.model_name = "DQN-CartPole-v1"
    def train(self):
        self.model.learn(total_timesteps=int(2e6))
        self.model.save(self.model_name)

    def evaluate(self):
        eval_env = Monitor(gym.make("CartPole-v1", render_mode='rgb_array'))
        mean_reward, std_reward = evaluate_policy(self.model, eval_env, n_eval_episodes=10, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


    def main(self):
        self.train()
        self.evaluate()

if __name__ == "__main__":
    agent = DQNAgentSB3(gym.make("CartPole-v1"))
    agent.main()