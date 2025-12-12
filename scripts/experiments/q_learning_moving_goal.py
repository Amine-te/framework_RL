"""
Q-Learning with Moving Goals: State Representation Comparison
==============================================================

This script compares two Q-Learning agents on a grid where the goal
location changes randomly each episode:
1. Goal-Aware Agent: State includes both position and goal location
2. Position-Only Agent: State only includes current position

Metrics compared:
- Average return per episode
- Average Q-value change (learning dynamics)
- Success rate
- Convergence behavior

Usage:
------
python q_learning_moving_goal.py --episodes 5000
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gridworld import GridWorldEnv
from scripts.utils.helpers import moving_average, get_random_goal

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class GoalAwareAgent:
    """
    Q-Learning agent with goal position in state representation.
    State: (agent_row, agent_col, goal_row, goal_col)
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
        
        # Track Q-value changes for analysis
        self.q_value_changes = []
    
    def get_state_key(self, agent_pos, goal_pos):
        """Convert observation and goal to hashable state key."""
        return (agent_pos[0], agent_pos[1], goal_pos[0], goal_pos[1])
    
    def choose_action(self, agent_pos, goal_pos):
        """Epsilon-greedy action selection."""
        state = self.get_state_key(agent_pos, goal_pos)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        # Exploitation: choose action with highest Q-value
        q_values = [self.q_table[state][a] for a in range(self.num_actions)]
        max_q = max(q_values)
        
        # Handle ties by random selection
        best_actions = [a for a in range(self.num_actions) if q_values[a] == max_q]
        return np.random.choice(best_actions)
    
    def update(self, agent_pos, goal_pos, action, reward, next_agent_pos, next_goal_pos, done):
        """Update Q-table using Q-learning update rule."""
        state = self.get_state_key(agent_pos, goal_pos)
        next_state = self.get_state_key(next_agent_pos, next_goal_pos)
        
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
        
        # Track Q-value change
        q_change = abs(new_q - current_q)
        self.q_value_changes.append(q_change)
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_q_changes(self):
        """Reset Q-value change tracker."""
        self.q_value_changes = []
    
    def get_avg_q_change(self):
        """Get average Q-value change."""
        if len(self.q_value_changes) == 0:
            return 0.0
        return np.mean(self.q_value_changes)


class PositionOnlyAgent:
    """
    Q-Learning agent with only position in state representation.
    State: (agent_row, agent_col)
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
        
        # Track Q-value changes for analysis
        self.q_value_changes = []
    
    def get_state_key(self, agent_pos):
        """Convert observation to hashable state key (position only)."""
        return (agent_pos[0], agent_pos[1])
    
    def choose_action(self, agent_pos, goal_pos=None):
        """Epsilon-greedy action selection (goal_pos ignored)."""
        state = self.get_state_key(agent_pos)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        # Exploitation: choose action with highest Q-value
        q_values = [self.q_table[state][a] for a in range(self.num_actions)]
        max_q = max(q_values)
        
        # Handle ties by random selection
        best_actions = [a for a in range(self.num_actions) if q_values[a] == max_q]
        return np.random.choice(best_actions)
    
    def update(self, agent_pos, goal_pos, action, reward, next_agent_pos, next_goal_pos, done):
        """Update Q-table using Q-learning update rule (goal positions ignored)."""
        state = self.get_state_key(agent_pos)
        next_state = self.get_state_key(next_agent_pos)
        
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
        
        # Track Q-value change
        q_change = abs(new_q - current_q)
        self.q_value_changes.append(q_change)
        
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_q_changes(self):
        """Reset Q-value change tracker."""
        self.q_value_changes = []
    
    def get_avg_q_change(self):
        """Get average Q-value change."""
        if len(self.q_value_changes) == 0:
            return 0.0
        return np.mean(self.q_value_changes)


# get_random_goal is now imported from scripts.utils.helpers

def train_agent_moving_goal(env, agent, episodes, grid_size, verbose=True):
    """
    Train an agent where the goal changes each episode.
    
    Returns:
        rewards_history: List of total rewards per episode
        steps_history: List of steps taken per episode
        success_history: List of success indicators per episode
        q_change_history: List of average Q-value changes per episode
    """
    rewards_history = []
    steps_history = []
    success_history = []
    q_change_history = []
    
    obstacles = [[2, 2], [2, 3], [4, 4]]  # Fixed obstacles
    
    for episode in range(episodes):
        # Generate new random goal for this episode
        goal = get_random_goal(grid_size, avoid_positions=obstacles + [[0, 0]])
        
        # Update environment with new goal
        env.goals = [goal]
        
        # Reset Q-value change tracker for this episode
        agent.reset_q_changes()
        
        observation, info = env.reset(seed=RANDOM_SEED + episode)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(observation, goal)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            agent.update(observation, goal, action, reward, next_observation, goal, done)
            
            observation = next_observation
            total_reward += reward
        
        agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        steps_history.append(env.episode_steps)
        success_history.append(1 if terminated else 0)
        q_change_history.append(agent.get_avg_q_change())
        
        if verbose and (episode + 1) % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            avg_q_change = np.mean(q_change_history[-100:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Success: {success_rate:.1f}% | "
                  f"Avg Q-Change: {avg_q_change:.4f} | "
                  f"ε: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history, success_history, q_change_history


# moving_average is now imported from scripts.utils.helpers


def run_comparison(episodes=5000, grid_size=7):
    """Run comparison between goal-aware and position-only agents."""
    
    import os
    
    print("=" * 80)
    print("Q-LEARNING WITH MOVING GOALS: STATE REPRESENTATION COMPARISON")
    print("=" * 80)
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Episodes: {episodes}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Goal Location: Changes randomly each episode")
    print("=" * 80)
    
    # Create environment (goals will be updated each episode)
    env = GridWorldEnv(
        grid_size=grid_size,
        goals=[[grid_size-1, grid_size-1]],  # Initial goal (will change)
        start_pos=[0, 0],
        obstacles=[[2, 2], [2, 3], [4, 4]],
        reward_goal=10.0,
        reward_step=-0.1,
        max_steps=100
    )
    
    # Train Goal-Aware Agent
    print("\n" + "-" * 80)
    print("TRAINING GOAL-AWARE AGENT (State includes goal position)")
    print("-" * 80)
    goal_aware_agent = GoalAwareAgent(
        grid_size=grid_size,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    ga_rewards, ga_steps, ga_success, ga_q_changes = train_agent_moving_goal(
        env, goal_aware_agent, episodes, grid_size, verbose=True
    )
    
    # Train Position-Only Agent
    print("\n" + "-" * 80)
    print("TRAINING POSITION-ONLY AGENT (State only includes position)")
    print("-" * 80)
    position_agent = PositionOnlyAgent(
        grid_size=grid_size,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    po_rewards, po_steps, po_success, po_q_changes = train_agent_moving_goal(
        env, position_agent, episodes, grid_size, verbose=True
    )
    
    env.close()
    
    # Summary Statistics
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Last 500 Episodes)")
    print("=" * 80)
    
    ga_final_reward = np.mean(ga_rewards[-500:])
    po_final_reward = np.mean(po_rewards[-500:])
    ga_final_steps = np.mean(ga_steps[-500:])
    po_final_steps = np.mean(po_steps[-500:])
    ga_final_success = np.mean(ga_success[-500:]) * 100
    po_final_success = np.mean(po_success[-500:]) * 100
    ga_final_q_change = np.mean(ga_q_changes[-500:])
    po_final_q_change = np.mean(po_q_changes[-500:])
    
    print(f"\nGoal-Aware Agent:")
    print(f"  Average Return:    {ga_final_reward:.3f}")
    print(f"  Average Steps:     {ga_final_steps:.2f}")
    print(f"  Success Rate:      {ga_final_success:.2f}%")
    print(f"  Avg Q-Change:      {ga_final_q_change:.5f}")
    print(f"  Q-Table Size:      {len(goal_aware_agent.q_table)} states")
    
    print(f"\nPosition-Only Agent:")
    print(f"  Average Return:    {po_final_reward:.3f}")
    print(f"  Average Steps:     {po_final_steps:.2f}")
    print(f"  Success Rate:      {po_final_success:.2f}%")
    print(f"  Avg Q-Change:      {po_final_q_change:.5f}")
    print(f"  Q-Table Size:      {len(position_agent.q_table)} states")
    
    print(f"\nPerformance Improvement (Goal-Aware vs Position-Only):")
    reward_improvement = ((ga_final_reward - po_final_reward) / abs(po_final_reward)) * 100
    success_improvement = ga_final_success - po_final_success
    print(f"  Return Improvement:  {reward_improvement:+.2f}%")
    print(f"  Success Rate Diff:   {success_improvement:+.2f} percentage points")
    
    # Generate Plots
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning with Moving Goals: State Representation Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Average Return
    window = 100
    ga_rewards_smooth = moving_average(ga_rewards, window)
    po_rewards_smooth = moving_average(po_rewards, window)
    
    axes[0, 0].plot(range(len(ga_rewards_smooth)), ga_rewards_smooth, 
                    label='Goal-Aware', linewidth=2, color='#2E86AB')
    axes[0, 0].plot(range(len(po_rewards_smooth)), po_rewards_smooth, 
                    label='Position-Only', linewidth=2, color='#A23B72')
    axes[0, 0].set_title('Average Return per Episode', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Episode', fontsize=10)
    axes[0, 0].set_ylabel('Return (smoothed)', fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average Q-Value Change
    ga_q_smooth = moving_average(ga_q_changes, window)
    po_q_smooth = moving_average(po_q_changes, window)
    
    axes[0, 1].plot(range(len(ga_q_smooth)), ga_q_smooth, 
                    label='Goal-Aware', linewidth=2, color='#2E86AB')
    axes[0, 1].plot(range(len(po_q_smooth)), po_q_smooth, 
                    label='Position-Only', linewidth=2, color='#A23B72')
    axes[0, 1].set_title('Average Q-Value Change per Episode', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Episode', fontsize=10)
    axes[0, 1].set_ylabel('Avg |ΔQ| (smoothed)', fontsize=10)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    ga_success_smooth = moving_average(ga_success, window)
    po_success_smooth = moving_average(po_success, window)
    
    axes[1, 0].plot(range(len(ga_success_smooth)), ga_success_smooth, 
                    label='Goal-Aware', linewidth=2, color='#2E86AB')
    axes[1, 0].plot(range(len(po_success_smooth)), po_success_smooth, 
                    label='Position-Only', linewidth=2, color='#A23B72')
    axes[1, 0].set_title('Success Rate', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Episode', fontsize=10)
    axes[1, 0].set_ylabel('Success Rate (smoothed)', fontsize=10)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average Steps
    ga_steps_smooth = moving_average(ga_steps, window)
    po_steps_smooth = moving_average(po_steps, window)
    
    axes[1, 1].plot(range(len(ga_steps_smooth)), ga_steps_smooth, 
                    label='Goal-Aware', linewidth=2, color='#2E86AB')
    axes[1, 1].plot(range(len(po_steps_smooth)), po_steps_smooth, 
                    label='Position-Only', linewidth=2, color='#A23B72')
    axes[1, 1].set_title('Average Steps per Episode', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Episode', fontsize=10)
    axes[1, 1].set_ylabel('Steps (smoothed)', fontsize=10)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f'q_learning_moving_goal_ep{episodes}_grid{grid_size}_comparison.png'
    plt.savefig(str(results_dir / filename), dpi=300, bbox_inches='tight')
    print(f"Saved: {results_dir / filename}")
    plt.show()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Q-Learning with Moving Goals: Compare state representations'
    )
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes (default: 5000)')
    parser.add_argument('--grid_size', type=int, default=7,
                        help='Size of the grid (default: 7)')
    
    args = parser.parse_args()
    
    run_comparison(episodes=args.episodes, grid_size=args.grid_size)