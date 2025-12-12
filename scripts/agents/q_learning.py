"""
Q-Learning Agent (Tabular)
===========================

A model-free, off-policy TD algorithm that learns optimal Q-values.
One of the most popular RL algorithms!

Pros: No environment model needed, learns from individual transitions, fast
Cons: Can overestimate values, requires careful hyperparameter tuning
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """Q-Learning Agent for GridWorld (Tabular)"""
    
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        Args:
            env: GridWorld environment
            gamma: Discount factor
            alpha: Learning rate
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(4))
        
        self.name = "Q-Learning"
    
    def get_action(self, observation, training=True):
        """
        Get action using epsilon-greedy policy
        
        Args:
            observation: Current state [row, col]
            training: If True, use exploration; if False, use greedy policy
        """
        state = tuple(observation)
        
        # Epsilon-greedy exploration during training
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 4)
        else:
            return np.argmax(self.Q[state])
    
    def train(self, num_episodes=1000, seed=42):
        """
        Train the agent using Q-Learning
        
        Args:
            num_episodes: Number of episodes to train
            seed: Random seed
            
        Returns:
            Training statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {self.name}")
        print(f"{'='*60}")
        
        np.random.seed(seed)
        
        episode_returns = []
        episode_lengths = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, info = self.env.reset(seed=seed + episode)
            state = tuple(obs)
            total_reward = 0
            done = False
            
            while not done:
                # Choose action
                action = self.get_action(obs, training=True)
                
                # Take action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_state = tuple(next_obs)
                total_reward += reward
                done = terminated or truncated
                
                # Q-Learning update
                if terminated:
                    # Terminal state: no future rewards
                    td_target = reward
                else:
                    # TD target: reward + discounted max future Q-value
                    td_target = reward + self.gamma * np.max(self.Q[next_state])
                
                # Update Q-value
                td_error = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_error
                
                # Move to next state
                state = next_state
                obs = next_obs
            
            # Track statistics
            episode_returns.append(total_reward)
            episode_lengths.append(self.env.episode_steps)
            
            if terminated:
                success_count += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Progress update
            if (episode + 1) % 100 == 0:
                recent_success = np.mean([episode_returns[i] > 0 for i in range(max(0, episode - 99), episode + 1)]) * 100
                avg_return = np.mean(episode_returns[max(0, episode - 99):episode + 1])
                print(f"Episode {episode + 1}: Avg Return = {avg_return:.2f}, Success Rate = {recent_success:.1f}%, Epsilon = {self.epsilon:.3f}")
        
        return {
            'episode_returns': episode_returns,
            'episode_lengths': episode_lengths,
            'success_rate': (success_count / num_episodes) * 100,
            'final_epsilon': self.epsilon
        }
    
    def evaluate(self, num_episodes=100, seed=42):
        """Evaluate the learned policy"""
        returns = []
        lengths = []
        successes = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset(seed=seed + episode)
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(obs, training=False)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            returns.append(total_reward)
            lengths.append(self.env.episode_steps)
            successes.append(terminated)
        
        return {
            'returns': returns,
            'lengths': lengths,
            'successes': successes,
            'success_rate': np.mean(successes) * 100,
            'avg_return': np.mean(returns),
            'avg_length': np.mean(lengths)
        }

