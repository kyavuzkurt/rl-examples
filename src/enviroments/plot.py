import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self):
        self.fig_size = (12, 5)
        self.reward_window = 50  # Window size for reward smoothing
        
    def plot_training_results(self, rewards, epsilon_values, save_path=None):
        """
        Plot training results including smoothed rewards and epsilon decay
        
        Args:
            rewards (list): List of episode rewards
            epsilon_values (list): List of epsilon values during training
            save_path (str, optional): Path to save the plot. If None, displays the plot
        """
        plt.figure(figsize=self.fig_size)
        
        # Plot smoothed rewards
        plt.subplot(1, 2, 1)
        smoothed_rewards = self.smooth_rewards(rewards)
        episodes = range(len(smoothed_rewards))
        
        plt.plot(episodes, smoothed_rewards, 'b-', label='Smoothed Reward')
        plt.plot(rewards, 'b.', alpha=0.1, label='Raw Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot epsilon decay
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_values, 'r-', label='Epsilon Value')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def plot_evaluation_results(self, rewards, save_path=None):
        """
        Plot evaluation results including reward distribution
        
        Args:
            rewards (list): List of evaluation episode rewards
            save_path (str, optional): Path to save the plot. If None, displays the plot
        """
        plt.figure(figsize=(8, 6))
        
        plt.boxplot(rewards)
        plt.scatter(np.ones_like(rewards), rewards, alpha=0.5)
        
        plt.title('Evaluation Results')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        
        # Add any additional plotting logic here if needed
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def smooth_rewards(self, rewards):
        """
        Apply moving average smoothing to the rewards
        
        Args:
            rewards (list): List of raw rewards
            
        Returns:
            numpy.array: Smoothed rewards
        """
        return np.convolve(rewards, 
                          np.ones(self.reward_window)/self.reward_window, 
                          mode='valid')
