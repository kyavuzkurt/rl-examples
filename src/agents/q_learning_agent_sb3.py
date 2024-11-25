import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class QLearningAgentSB3:
    def __init__(self, env):
        self.env = env
        self.model = PPO(policy="MlpPolicy",
                         env=self.env,
                         verbose=1,
                         n_steps=1024,
                         batch_size=64,
                         gamma=0.99,
                         gae_lambda=0.95,
                         ent_coef=0.01,
                         max_grad_norm=0.5)
        self.model_name = "PPO-CartPole-v1"

    def train(self):
        self.model.learn(total_timesteps=int(1e6))
        self.model.save(self.model_name)

    def evaluate(self):
        eval_env = Monitor(gym.make("CartPole-v1", render_mode='human'))
        mean_reward, std_reward = evaluate_policy(self.model, eval_env, n_eval_episodes=10, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    def main(self):
        #self.train()
        self.model = self.model.load(self.model_name)
        self.evaluate()
        
if __name__ == "__main__":
    agent = QLearningAgentSB3(gym.make("CartPole-v1"))
    agent.main()