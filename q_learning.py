"""
Q-Learning Parameter Study with Obstacles
==========================================

This script trains Q-Learning agents on different grid configurations
with varying hyperparameters and visualizes convergence behavior.

Features:
- Multiple grid sizes (5x5, 7x7, 10x10)
- Obstacles in the environment
- Parameter exploration (alpha, gamma)
- Convergence visualization
- Agent persistence (saved to 'agents' directory)

Usage:
------
python q_learning.py --episodes 2000
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from collections import defaultdict
from gridworld import GridWorldEnv

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class QLearningAgent:
    """
    Q-Learning Agent for GridWorld environments.
    
    Parameters:
        grid_size (int): Size of the grid
        alpha (float): Learning rate
        gamma (float): Discount factor
        epsilon (float): Exploration rate
        epsilon_decay (float): Decay rate for epsilon
        epsilon_min (float): Minimum epsilon value
    """
    
    def __init__(self, grid_size, alpha=0.1, gamma=0.9, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: nested dict {state: {action: value}}
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.num_actions = 4  # UP, RIGHT, DOWN, LEFT
    
    def get_state_key(self, observation):
        """Convert observation to hashable state key."""
        return tuple(observation)
    
    def choose_action(self, observation):
        """Epsilon-greedy action selection."""
        state = self.get_state_key(observation)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        # Exploitation: choose action with highest Q-value
        q_values = [self.q_table[state][a] for a in range(self.num_actions)]
        max_q = max(q_values)
        
        # Handle ties by random selection
        best_actions = [a for a in range(self.num_actions) if q_values[a] == max_q]
        return np.random.choice(best_actions)
    
    def update(self, observation, action, reward, next_observation, done):
        """Update Q-table using Q-learning update rule."""
        state = self.get_state_key(observation)
        next_state = self.get_state_key(next_observation)
        
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Maximum Q-value for next state
        if done:
            max_next_q = 0.0
        else:
            next_q_values = [self.q_table[next_state][a] for a in range(self.num_actions)]
            max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save agent to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'grid_size': self.grid_size
            }, f)
        print(f"Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        agent = cls(
            grid_size=data['grid_size'],
            alpha=data['alpha'],
            gamma=data['gamma'],
            epsilon=data['epsilon']
        )
        agent.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
        return agent


def train_agent(env, agent, episodes, verbose=True):
    """
    Train a Q-learning agent.
    
    Returns:
        rewards_history: List of total rewards per episode
        steps_history: List of steps taken per episode
        success_history: List of success indicators per episode
    """
    rewards_history = []
    steps_history = []
    success_history = []
    
    for episode in range(episodes):
        observation, info = env.reset(seed=RANDOM_SEED + episode)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            agent.update(observation, action, reward, next_observation, done)
            
            observation = next_observation
            total_reward += reward
        
        agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        steps_history.append(env.episode_steps)
        success_history.append(1 if terminated else 0)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history, success_history


def moving_average(data, window=100):
    """Calculate moving average for smoothing."""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def create_environment(grid_size):
    """Create a GridWorld environment with obstacles based on grid size."""
    if grid_size == 5:
        return GridWorldEnv(
            grid_size=5,
            goals=[[4, 4]],
            start_pos=[0, 0],
            obstacles=[[2, 2], [2, 3]],
            reward_goal=10.0,
            reward_step=-0.1,
            max_steps=50
        )
    elif grid_size == 7:
        return GridWorldEnv(
            grid_size=7,
            goals=[[6, 6]],
            start_pos=[0, 0],
            obstacles=[[2, 2], [2, 3], [2, 4], [4, 4], [4, 5]],
            reward_goal=10.0,
            reward_step=-0.1,
            max_steps=75
        )
    elif grid_size == 10:
        return GridWorldEnv(
            grid_size=10,
            goals=[[9, 9]],
            start_pos=[0, 0],
            obstacles=[[3, i] for i in range(5)] + [[7, i] for i in range(5, 10)],
            reward_goal=10.0,
            reward_step=-0.1,
            max_steps=150
        )
    else:
        raise ValueError(f"Unsupported grid size: {grid_size}")


def run_parameter_study(episodes=2000):
    """Run parameter study and generate comparison plots."""
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('agents', exist_ok=True)
    
    print("=" * 70)
    print("Q-LEARNING PARAMETER STUDY")
    print("=" * 70)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Episodes per configuration: {episodes}\n")
    
    # Study 1: Alpha (Learning Rate) Comparison
    print("\n" + "=" * 70)
    print("STUDY 1: Learning Rate (Alpha) Comparison")
    print("=" * 70)
    
    alphas = [0.05, 0.1, 0.3, 0.5]
    alpha_results = {}
    
    env = create_environment(7)
    
    for alpha in alphas:
        print(f"\nTraining with alpha={alpha}")
        agent = QLearningAgent(grid_size=7, alpha=alpha, gamma=0.9)
        rewards, steps, success = train_agent(env, agent, episodes)
        alpha_results[alpha] = {'rewards': rewards, 'steps': steps, 'success': success}
        agent.save(f'agents/agent_alpha_{alpha}.pkl')
    
    env.close()
    
    # Study 2: Gamma (Discount Factor) Comparison
    print("\n" + "=" * 70)
    print("STUDY 2: Discount Factor (Gamma) Comparison")
    print("=" * 70)
    
    gammas = [0.7, 0.9, 0.95, 0.99]
    gamma_results = {}
    
    env = create_environment(7)
    
    for gamma in gammas:
        print(f"\nTraining with gamma={gamma}")
        agent = QLearningAgent(grid_size=7, alpha=0.1, gamma=gamma)
        rewards, steps, success = train_agent(env, agent, episodes)
        gamma_results[gamma] = {'rewards': rewards, 'steps': steps, 'success': success}
        agent.save(f'agents/agent_gamma_{gamma}.pkl')
    
    env.close()
    
    # Study 3: Grid Size Comparison
    print("\n" + "=" * 70)
    print("STUDY 3: Grid Size Comparison")
    print("=" * 70)
    
    grid_sizes = [5, 7, 10]
    grid_results = {}
    
    for size in grid_sizes:
        print(f"\nTraining on {size}x{size} grid")
        env = create_environment(size)
        agent = QLearningAgent(grid_size=size, alpha=0.1, gamma=0.9)
        rewards, steps, success = train_agent(env, agent, episodes)
        grid_results[size] = {'rewards': rewards, 'steps': steps, 'success': success}
        agent.save(f'agents/agent_grid_{size}.pkl')
        env.close()
    
    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    # Plot 1: Alpha Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Q-Learning: Learning Rate (Alpha) Comparison', fontsize=16, fontweight='bold')
    
    for alpha in alphas:
        smoothed_rewards = moving_average(alpha_results[alpha]['rewards'])
        smoothed_steps = moving_average(alpha_results[alpha]['steps'])
        smoothed_success = moving_average(alpha_results[alpha]['success'])
        
        axes[0].plot(smoothed_rewards, label=f'α={alpha}', linewidth=2)
        axes[1].plot(smoothed_steps, label=f'α={alpha}', linewidth=2)
        axes[2].plot(smoothed_success, label=f'α={alpha}', linewidth=2)
    
    axes[0].set_title('Cumulative Return per Episode', fontweight='bold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return (smoothed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Average Steps', fontweight='bold')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps (smoothed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Success Rate', fontweight='bold')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate (smoothed)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/alpha_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/alpha_comparison.png")
    plt.close()
    
    # Plot 2: Gamma Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Q-Learning: Discount Factor (Gamma) Comparison', fontsize=16, fontweight='bold')
    
    for gamma in gammas:
        smoothed_rewards = moving_average(gamma_results[gamma]['rewards'])
        smoothed_steps = moving_average(gamma_results[gamma]['steps'])
        smoothed_success = moving_average(gamma_results[gamma]['success'])
        
        axes[0].plot(smoothed_rewards, label=f'γ={gamma}', linewidth=2)
        axes[1].plot(smoothed_steps, label=f'γ={gamma}', linewidth=2)
        axes[2].plot(smoothed_success, label=f'γ={gamma}', linewidth=2)
    
    axes[0].set_title('Cumulative Return per Episode', fontweight='bold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return (smoothed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Average Steps', fontweight='bold')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps (smoothed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Success Rate', fontweight='bold')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate (smoothed)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/gamma_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/gamma_comparison.png")
    plt.close()
    
    # Plot 3: Grid Size Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Q-Learning: Grid Size Comparison', fontsize=16, fontweight='bold')
    
    for size in grid_sizes:
        smoothed_rewards = moving_average(grid_results[size]['rewards'])
        smoothed_steps = moving_average(grid_results[size]['steps'])
        smoothed_success = moving_average(grid_results[size]['success'])
        
        axes[0].plot(smoothed_rewards, label=f'{size}x{size}', linewidth=2)
        axes[1].plot(smoothed_steps, label=f'{size}x{size}', linewidth=2)
        axes[2].plot(smoothed_success, label=f'{size}x{size}', linewidth=2)
    
    axes[0].set_title('Cumulative Return per Episode', fontweight='bold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return (smoothed)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Average Steps', fontweight='bold')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps (smoothed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Success Rate', fontweight='bold')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate (smoothed)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/grid_size_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/grid_size_comparison.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("PARAMETER STUDY COMPLETE")
    print("=" * 70)
    print(f"All plots saved to 'plots/' directory")
    print(f"All trained agents saved to 'agents/' directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning Parameter Study')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes per training run (default: 2000)')
    
    args = parser.parse_args()
    
    run_parameter_study(episodes=args.episodes)