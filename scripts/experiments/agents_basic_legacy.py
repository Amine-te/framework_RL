"""
Reinforcement Learning Agents for GridWorld
============================================

This module implements 4 classic RL algorithms:
1. Policy Iteration (PI) - Model-based, computes optimal policy iteratively
2. Value Iteration (VI) - Model-based, computes optimal value function
3. Monte Carlo (MC) - Model-free, learns from complete episodes
4. Q-Learning (QL) - Model-free, learns from individual transitions

Usage:
------
# Run a single agent
python agents_basic.py --agent PI

# Run all agents and compare
python agents_basic.py --agent all

# Run specific agents
python agents_basic.py --agent PI VI

# Custom episodes
python agents_basic.py --agent QL --episodes 2000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import os
from src.gridworld import GridWorldEnv


class PolicyIterationAgent:
    """
    Policy Iteration Agent
    ======================
    
    A model-based algorithm that alternates between:
    1. Policy Evaluation: Calculate value of current policy
    2. Policy Improvement: Update policy to be greedy w.r.t. values
    
    Pros: Guaranteed to find optimal policy, fast convergence
    Cons: Requires complete knowledge of environment dynamics
    
    How it works:
    - Iteratively evaluates and improves policy until convergence
    - Uses dynamic programming with full environment model
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6):
        """
        Args:
            env: GridWorld environment
            gamma: Discount factor (how much to value future rewards)
            theta: Convergence threshold for policy evaluation
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.grid_size = env.grid_size
        
        # Initialize value function and policy
        self.V = np.zeros((self.grid_size, self.grid_size))
        self.policy = np.random.randint(0, 4, (self.grid_size, self.grid_size))
        
        self.name = "Policy Iteration"
    
    def _get_next_state(self, state, action):
        """Simulate taking an action from a state"""
        row, col = state
        new_row, new_col = row, col
        
        if action == 0:  # UP
            new_row -= 1
        elif action == 1:  # RIGHT
            new_col += 1
        elif action == 2:  # DOWN
            new_row += 1
        elif action == 3:  # LEFT
            new_col -= 1
        
        # Check boundaries
        new_row = max(0, min(self.grid_size - 1, new_row))
        new_col = max(0, min(self.grid_size - 1, new_col))
        
        # Check obstacles
        if [new_row, new_col] in self.env.obstacles:
            return state  # Stay in place if obstacle
        
        return (new_row, new_col)
    
    def _get_reward(self, state, action, next_state):
        """Get reward for transition"""
        if list(next_state) in self.env.goals:
            return self.env.reward_goal
        return self.env.reward_step
    
    def _is_goal_state(self, state):
        """Check if state is a goal state"""
        return list(state) in self.env.goals
    
    def _policy_evaluation(self):
        """Evaluate current policy until convergence"""
        while True:
            delta = 0
            new_V = np.copy(self.V)
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = (row, col)
                    
                    # Skip if goal state
                    if self._is_goal_state(state):
                        continue
                    
                    # Get action from current policy
                    action = self.policy[row, col]
                    
                    # Calculate expected value
                    next_state = self._get_next_state(state, action)
                    reward = self._get_reward(state, action, next_state)
                    
                    new_V[row, col] = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    
                    delta = max(delta, abs(new_V[row, col] - self.V[row, col]))
            
            self.V = new_V
            
            if delta < self.theta:
                break
    
    def _policy_improvement(self):
        """Improve policy by being greedy w.r.t. value function"""
        policy_stable = True
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = (row, col)
                
                # Skip if goal state
                if self._is_goal_state(state):
                    continue
                
                old_action = self.policy[row, col]
                
                # Find best action
                action_values = []
                for action in range(4):
                    next_state = self._get_next_state(state, action)
                    reward = self._get_reward(state, action, next_state)
                    value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    action_values.append(value)
                
                best_action = np.argmax(action_values)
                self.policy[row, col] = best_action
                
                if old_action != best_action:
                    policy_stable = False
        
        return policy_stable
    
    def train(self, max_iterations=100):
        """
        Train the agent using Policy Iteration
        
        Args:
            max_iterations: Maximum number of policy iteration loops
        
        Returns:
            Training statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {self.name}")
        print(f"{'='*60}")
        
        for iteration in range(max_iterations):
            # Evaluate current policy
            self._policy_evaluation()
            
            # Improve policy
            policy_stable = self._policy_improvement()
            
            print(f"Iteration {iteration + 1}: Policy {'converged!' if policy_stable else 'updated'}")
            
            if policy_stable:
                print(f"Policy converged after {iteration + 1} iterations!")
                break
        
        return {'iterations': iteration + 1, 'converged': policy_stable}
    
    def get_action(self, observation):
        """Get action for a given observation"""
        row, col = observation
        return self.policy[row, col]
    
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
                action = self.get_action(obs)
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


class ValueIterationAgent:
    """
    Value Iteration Agent
    =====================
    
    A model-based algorithm that directly computes the optimal value function
    by iteratively applying the Bellman optimality equation.
    
    Pros: Often faster than Policy Iteration, simpler implementation
    Cons: Requires complete knowledge of environment dynamics
    
    How it works:
    - Repeatedly updates value of each state by considering all possible actions
    - Extracts optimal policy once values have converged
    """
    
    def __init__(self, env, gamma=0.99, theta=1e-6):
        """
        Args:
            env: GridWorld environment
            gamma: Discount factor
            theta: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.grid_size = env.grid_size
        
        # Initialize value function
        self.V = np.zeros((self.grid_size, self.grid_size))
        self.policy = None
        
        self.name = "Value Iteration"
    
    def _get_next_state(self, state, action):
        """Simulate taking an action from a state"""
        row, col = state
        new_row, new_col = row, col
        
        if action == 0:  # UP
            new_row -= 1
        elif action == 1:  # RIGHT
            new_col += 1
        elif action == 2:  # DOWN
            new_row += 1
        elif action == 3:  # LEFT
            new_col -= 1
        
        # Check boundaries
        new_row = max(0, min(self.grid_size - 1, new_row))
        new_col = max(0, min(self.grid_size - 1, new_col))
        
        # Check obstacles
        if [new_row, new_col] in self.env.obstacles:
            return state  # Stay in place if obstacle
        
        return (new_row, new_col)
    
    def _get_reward(self, state, action, next_state):
        """Get reward for transition"""
        if list(next_state) in self.env.goals:
            return self.env.reward_goal
        return self.env.reward_step
    
    def _is_goal_state(self, state):
        """Check if state is a goal state"""
        return list(state) in self.env.goals
    
    def train(self, max_iterations=1000):
        """
        Train the agent using Value Iteration
        
        Args:
            max_iterations: Maximum number of iterations
        
        Returns:
            Training statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {self.name}")
        print(f"{'='*60}")
        
        for iteration in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = (row, col)
                    
                    # Skip if goal state
                    if self._is_goal_state(state):
                        continue
                    
                    # Calculate value for each action
                    action_values = []
                    for action in range(4):
                        next_state = self._get_next_state(state, action)
                        reward = self._get_reward(state, action, next_state)
                        value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                        action_values.append(value)
                    
                    # Take maximum value
                    new_V[row, col] = max(action_values)
                    delta = max(delta, abs(new_V[row, col] - self.V[row, col]))
            
            self.V = new_V
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Delta = {delta:.6f}")
            
            if delta < self.theta:
                print(f"Converged after {iteration + 1} iterations!")
                break
        
        # Extract policy from value function
        self._extract_policy()
        
        return {'iterations': iteration + 1, 'delta': delta}
    
    def _extract_policy(self):
        """Extract optimal policy from value function"""
        self.policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = (row, col)
                
                if self._is_goal_state(state):
                    continue
                
                # Find best action
                action_values = []
                for action in range(4):
                    next_state = self._get_next_state(state, action)
                    reward = self._get_reward(state, action, next_state)
                    value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    action_values.append(value)
                
                self.policy[row, col] = np.argmax(action_values)
    
    def get_action(self, observation):
        """Get action for a given observation"""
        row, col = observation
        return self.policy[row, col]
    
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
                action = self.get_action(obs)
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


class MonteCarloAgent:
    """
    Monte Carlo Agent
    =================
    
    A model-free algorithm that learns from complete episodes.
    Uses first-visit MC to estimate action values.
    
    Pros: No environment model needed, learns from experience
    Cons: Requires complete episodes, can be slow to converge
    
    How it works:
    - Follows epsilon-greedy policy to explore
    - Updates Q-values based on returns from complete episodes
    - Gradually reduces exploration (epsilon decay)
    """
    
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


class QLearningAgent:
    """
    Q-Learning Agent
    ================
    
    A model-free, off-policy TD algorithm that learns optimal Q-values.
    One of the most popular RL algorithms!
    
    Pros: No environment model needed, learns from individual transitions, fast
    Cons: Can overestimate values, requires careful hyperparameter tuning
    
    How it works:
    - Updates Q-values after each step (not waiting for episode end)
    - Uses TD target: reward + gamma * max(Q(next_state))
    - Gradually reduces exploration and learning rate
    """
    
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


def compare_agents(agents_results, save_dir=None, grid_size=None, episodes=None, agents_list=None):
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / 'results' / 'plots'
    save_dir = Path(save_dir)
    """
    Create comparison plot for multiple agents and save to file
    
    Args:
        agents_results: Dictionary mapping agent names to their results
        save_dir: Directory to save the plot
        grid_size: Grid size for filename
        episodes: Number of episodes for filename
        agents_list: List of agent names for filename
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Agent Comparison on GridWorld', fontsize=16, fontweight='bold')
    
    # Prepare data
    agent_names = list(agents_results.keys())
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Cumulative Returns Comparison
    for i, name in enumerate(agent_names):
        if 'returns' in agents_results[name]:
            returns = agents_results[name]['returns']
            cumulative_returns = np.cumsum(returns)
            episodes_plot = np.arange(1, len(returns) + 1)
            ax.plot(episodes_plot, cumulative_returns, label=name, 
                   color=colors[i], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Cumulative Returns Over Episodes', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create meaningful filename
    filename_parts = ['agents_basic']
    if grid_size is not None:
        filename_parts.append(f'grid{grid_size}')
    if episodes is not None:
        filename_parts.append(f'ep{episodes}')
    if agents_list:
        agents_str = '_'.join([a.replace(' ', '_').replace('-', '_') for a in agents_list])
        filename_parts.append(agents_str)
    filename_parts.append('comparison')
    filename = '_'.join(filename_parts) + '.png'
    
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    return fig


def print_comparison_table(agents_results):
    """Print a formatted comparison table"""
    print("\n" + "="*80)
    print("AGENT COMPARISON TABLE")
    print("="*80)
    print(f"{'Agent':<20} {'Success Rate':<15} {'Avg Return':<15} {'Avg Length':<15}")
    print("-"*80)
    
    for name, results in agents_results.items():
        print(f"{name:<20} {results['success_rate']:>12.1f}%  {results['avg_return']:>12.2f}  {results['avg_length']:>12.1f}")
    
    print("="*80)


def main():
    """Main function to run agents individually or compare them"""
    
    parser = argparse.ArgumentParser(description='Train and evaluate RL agents on GridWorld')
    parser.add_argument('--agent', nargs='+', default=['all'],
                       choices=['PI', 'VI', 'MC', 'QL', 'all'],
                       help='Which agent(s) to run (default: all)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes for MC and QL (default: 1000)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--grid-size', type=int, default=5,
                       help='Size of the grid (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create environment with correct parameter names
    env = GridWorldEnv(
        grid_size=args.grid_size,
        goals=[[args.grid_size-1, args.grid_size-1]],
        start_pos=None,  # Random start
        reward_goal=10.0,
        reward_step=-1,
        max_steps=50
    )
    
    print("\n" + "="*80)
    print("GRIDWORLD REINFORCEMENT LEARNING AGENTS")
    print("="*80)
    print(f"Environment: {args.grid_size}x{args.grid_size} grid")
    print(f"Goals: {env.goals}")
    print(f"Training episodes (MC/QL): {args.episodes}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Determine which agents to run
    if 'all' in args.agent:
        agents_to_run = ['PI', 'VI', 'MC', 'QL']
    else:
        agents_to_run = args.agent
    
    # Dictionary to store results
    agents_results = {}
    
    # Train and evaluate each agent
    if 'PI' in agents_to_run:
        agent = PolicyIterationAgent(env)
        agent.train()
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        agents_results['Policy Iteration'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")

    if 'VI' in agents_to_run:
        agent = ValueIterationAgent(env)
        agent.train()
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        agents_results['Value Iteration'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")
    
    if 'MC' in agents_to_run:
        agent = MonteCarloAgent(env)
        train_results = agent.train(num_episodes=args.episodes, seed=args.seed)
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        # Include training data for learning curves
        eval_results['returns'] = train_results['episode_returns']
        agents_results['Monte Carlo'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")
    
    if 'QL' in agents_to_run:
        agent = QLearningAgent(env)
        train_results = agent.train(num_episodes=args.episodes, seed=args.seed)
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        # Include training data for learning curves
        eval_results['returns'] = train_results['episode_returns']
        agents_results['Q-Learning'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")
    
    # Print comparison table if multiple agents
    if len(agents_results) > 1:
        print_comparison_table(agents_results)
        
        # Create and save comparison plot
        fig = compare_agents(agents_results, 
                            grid_size=args.grid_size, 
                            episodes=args.episodes,
                            agents_list=agents_to_run)
        plt.close(fig)  # Close the figure to free memory
    elif len(agents_results) == 1:
        # Save plot even for single agent if it has training data
        name = list(agents_results.keys())[0]
        if 'returns' in agents_results[name]:
            fig = compare_agents(agents_results,
                                grid_size=args.grid_size,
                                episodes=args.episodes,
                                agents_list=agents_to_run)
            plt.close(fig)

if __name__ == "__main__":
    main()