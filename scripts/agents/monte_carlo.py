"""
Monte Carlo Agent
=================

A model-free algorithm that learns from complete episodes.
Uses first-visit MC to estimate action values.

Pros: No environment model needed, learns from experience
Cons: Requires complete episodes, can be slow to converge
"""

import numpy as np
from collections import defaultdict


class MonteCarloAgent:
    """Monte Carlo Agent for GridWorld"""
    
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        """
        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Q[state][action] = value
        self.Q = defaultdict(lambda: np.zeros(4))
        self.returns = defaultdict(list)  # For averaging returns
        
        self.name = "Monte Carlo"
    
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
        Train the agent using Monte Carlo learning
        
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
            # Generate episode
            episode_data = []
            obs, info = self.env.reset(seed=seed + episode)
            done = False
            
            while not done:
                state = tuple(obs)
                action = self.get_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_data.append((state, action, reward))
                obs = next_obs
                done = terminated or truncated
            
            # Calculate returns and update Q-values
            G = 0
            visited = set()
            
            # Process episode in reverse (backward view)
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = reward + self.gamma * G
                
                # First-visit MC: only update if state-action pair not seen earlier
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
            
            # Track statistics
            episode_return = sum([x[2] for x in episode_data])
            episode_returns.append(episode_return)
            episode_lengths.append(len(episode_data))
            
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

