"""
Function Approximation for Moving Goals: Linear FA & DQN vs Tabular Q-Learning
===============================================================================

Compares three agents on a grid where the goal location changes each episode:
1. Tabular Q-Learning (Position-Only): Uses Q-table with position state
2. Linear Function Approximation: Uses linear model with position features
3. Deep Q-Network (DQN): Uses neural network for Q-value approximation

All agents are "goal-blind" - they only see their position, not the goal location.

Usage:
------
python func_approx_comparison_no_goal_info.py --episodes 5000
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict, deque
from gridworld import GridWorldEnv
from q_learning_moving_goal import PositionOnlyAgent, get_random_goal, moving_average

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class LinearFAAgent:
    """
    Q-Learning agent using Linear Function Approximation.
    Features: position encoding (one-hot or tile coding)
    State: (agent_row, agent_col) - NO goal information
    """
    
    def __init__(self, grid_size, alpha=0.01, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, feature_type='onehot'):
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = 4
        
        self.feature_type = feature_type
        
        # Feature dimensions
        if feature_type == 'onehot':
            # One-hot encoding: grid_size^2 features
            self.feature_dim = grid_size * grid_size
        elif feature_type == 'coordinate':
            # Simple coordinates: 2 features (row, col) normalized
            self.feature_dim = 2
        elif feature_type == 'tile':
            # Tile coding: multiple overlapping tilings
            self.num_tilings = 4
            self.feature_dim = self.num_tilings * grid_size * grid_size
        
        # Weight matrix: [feature_dim, num_actions]
        self.weights = np.zeros((self.feature_dim, self.num_actions))
        
        # Track Q-value changes
        self.q_value_changes = []
    
    def get_features(self, agent_pos):
        """Extract features from agent position."""
        row, col = agent_pos
        
        if self.feature_type == 'onehot':
            # One-hot encoding of position
            features = np.zeros(self.feature_dim)
            idx = row * self.grid_size + col
            features[idx] = 1.0
            return features
        
        elif self.feature_type == 'coordinate':
            # Normalized coordinates
            return np.array([row / self.grid_size, col / self.grid_size])
        
        elif self.feature_type == 'tile':
            # Tile coding with offset tilings
            features = np.zeros(self.feature_dim)
            for t in range(self.num_tilings):
                offset = t / self.num_tilings
                tile_row = int((row + offset) % self.grid_size)
                tile_col = int((col + offset) % self.grid_size)
                idx = t * self.grid_size * self.grid_size + tile_row * self.grid_size + tile_col
                features[idx] = 1.0
            return features
    
    def get_q_values(self, agent_pos):
        """Compute Q-values for all actions."""
        features = self.get_features(agent_pos)
        return features @ self.weights  # [feature_dim] @ [feature_dim, 4] = [4]
    
    def choose_action(self, agent_pos, goal_pos=None):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        q_values = self.get_q_values(agent_pos)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def update(self, agent_pos, goal_pos, action, reward, next_agent_pos, next_goal_pos, done):
        """Update weights using gradient descent."""
        features = self.get_features(agent_pos)
        q_values = features @ self.weights
        current_q = q_values[action]
        
        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_agent_pos)
            target = reward + self.gamma * np.max(next_q_values)
        
        # Gradient descent update
        td_error = target - current_q
        self.weights[:, action] += self.alpha * td_error * features
        
        # Track Q-value change
        self.q_value_changes.append(abs(td_error))
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_q_changes(self):
        self.q_value_changes = []
    
    def get_avg_q_change(self):
        if len(self.q_value_changes) == 0:
            return 0.0
        return np.mean(self.q_value_changes)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    State: (agent_row, agent_col) - NO goal information
    """
    
    def __init__(self, grid_size, alpha=0.001, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, 
                 hidden_size=64, batch_size=32, buffer_size=10000,
                 target_update_freq=100):
        self.grid_size = grid_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = 4
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Input: 2 features (row, col normalized)
        self.input_dim = 2
        self.hidden_size = hidden_size
        
        # Simple 2-layer neural network (manual implementation)
        # Layer 1: input_dim -> hidden_size
        self.w1 = np.random.randn(self.input_dim, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        # Layer 2: hidden_size -> num_actions
        self.w2 = np.random.randn(hidden_size, self.num_actions) * 0.1
        self.b2 = np.zeros(self.num_actions)
        
        # Target network (frozen copy)
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Track updates
        self.update_count = 0
        self.q_value_changes = []
    
    def get_state_vector(self, agent_pos):
        """Convert position to normalized state vector."""
        return np.array([agent_pos[0] / self.grid_size, 
                        agent_pos[1] / self.grid_size])
    
    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)
    
    def forward(self, state, w1, b1, w2, b2):
        """Forward pass through network."""
        h = self.relu(state @ w1 + b1)
        q_values = h @ w2 + b2
        return q_values, h
    
    def get_q_values(self, agent_pos, use_target=False):
        """Get Q-values from network."""
        state = self.get_state_vector(agent_pos)
        if use_target:
            q_values, _ = self.forward(state, self.target_w1, self.target_b1,
                                      self.target_w2, self.target_b2)
        else:
            q_values, _ = self.forward(state, self.w1, self.b1, self.w2, self.b2)
        return q_values
    
    def choose_action(self, agent_pos, goal_pos=None):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        
        q_values = self.get_q_values(agent_pos)
        return np.argmax(q_values)
    
    def store_transition(self, agent_pos, action, reward, next_agent_pos, done):
        """Store experience in replay buffer."""
        state = self.get_state_vector(agent_pos)
        next_state = self.get_state_vector(next_agent_pos)
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update(self, agent_pos, goal_pos, action, reward, next_agent_pos, next_goal_pos, done):
        """Store transition and train on batch from replay buffer."""
        self.store_transition(agent_pos, action, reward, next_agent_pos, done)
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample random batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Forward pass on batch
        q_values_batch, hidden_batch = self.forward(states, self.w1, self.b1, self.w2, self.b2)
        
        # Target Q-values using target network
        next_q_values_batch, _ = self.forward(next_states, self.target_w1, self.target_b1,
                                              self.target_w2, self.target_b2)
        max_next_q = np.max(next_q_values_batch, axis=1)
        
        # Compute targets
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Current Q-values for taken actions
        current_q = q_values_batch[np.arange(self.batch_size), actions]
        
        # TD errors
        td_errors = targets - current_q
        self.q_value_changes.extend(np.abs(td_errors))
        
        # Backpropagation (manual)
        # Output layer gradient
        dq = np.zeros_like(q_values_batch)
        dq[np.arange(self.batch_size), actions] = -td_errors
        
        # Gradient for w2 and b2
        dw2 = hidden_batch.T @ dq / self.batch_size
        db2 = np.mean(dq, axis=0)
        
        # Gradient for hidden layer
        dhidden = dq @ self.w2.T
        dhidden[hidden_batch <= 0] = 0  # ReLU derivative
        
        # Gradient for w1 and b1
        dw1 = states.T @ dhidden / self.batch_size
        db1 = np.mean(dhidden, axis=0)
        
        # Update weights
        self.w1 -= self.alpha * dw1
        self.b1 -= self.alpha * db1
        self.w2 -= self.alpha * dw2
        self.b2 -= self.alpha * db2
        
        self.update_count += 1
        
        # Update target network periodically
        if self.update_count % self.target_update_freq == 0:
            self.target_w1 = self.w1.copy()
            self.target_b1 = self.b1.copy()
            self.target_w2 = self.w2.copy()
            self.target_b2 = self.b2.copy()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_q_changes(self):
        self.q_value_changes = []
    
    def get_avg_q_change(self):
        if len(self.q_value_changes) == 0:
            return 0.0
        return np.mean(self.q_value_changes)


def train_agent_moving_goal(env, agent, episodes, grid_size, agent_name, verbose=True):
    """Train an agent where the goal changes each episode."""
    rewards_history = []
    steps_history = []
    success_history = []
    q_change_history = []
    
    obstacles = [[2, 2], [2, 3], [4, 4]]
    
    for episode in range(episodes):
        goal = get_random_goal(grid_size, avoid_positions=obstacles + [[0, 0]])
        env.goals = [goal]
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
            print(f"[{agent_name}] Episode {episode + 1}/{episodes} | "
                  f"Reward: {avg_reward:.2f} | Steps: {avg_steps:.1f} | "
                  f"Success: {success_rate:.1f}% | Q-Δ: {avg_q_change:.4f} | "
                  f"ε: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history, success_history, q_change_history


def run_comparison(episodes=5000, grid_size=7):
    """Run comparison between tabular, linear FA, and DQN agents."""
    
    print("=" * 80)
    print("FUNCTION APPROXIMATION VS TABULAR Q-LEARNING")
    print("=" * 80)
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Episodes: {episodes}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Goal Location: Changes randomly each episode")
    print(f"State Representation: Position ONLY (goal-blind)")
    print("=" * 80)
    
    env = GridWorldEnv(
        grid_size=grid_size,
        goals=[[grid_size-1, grid_size-1]],
        start_pos=[0, 0],
        obstacles=[[2, 2], [2, 3], [4, 4]],
        reward_goal=10.0,
        reward_step=-0.1,
        max_steps=100
    )
    
    # Agent 1: Tabular Q-Learning
    print("\n" + "-" * 80)
    print("TRAINING AGENT 1: TABULAR Q-LEARNING")
    print("-" * 80)
    tabular_agent = PositionOnlyAgent(
        grid_size=grid_size,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    tab_rewards, tab_steps, tab_success, tab_q_changes = train_agent_moving_goal(
        env, tabular_agent, episodes, grid_size, "Tabular", verbose=True
    )
    
    # Agent 2: Linear Function Approximation
    print("\n" + "-" * 80)
    print("TRAINING AGENT 2: LINEAR FUNCTION APPROXIMATION")
    print("-" * 80)
    linear_agent = LinearFAAgent(
        grid_size=grid_size,
        alpha=0.01,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        feature_type='onehot'
    )
    lin_rewards, lin_steps, lin_success, lin_q_changes = train_agent_moving_goal(
        env, linear_agent, episodes, grid_size, "Linear-FA", verbose=True
    )
    
    # Agent 3: DQN
    print("\n" + "-" * 80)
    print("TRAINING AGENT 3: DEEP Q-NETWORK (DQN)")
    print("-" * 80)
    dqn_agent = DQNAgent(
        grid_size=grid_size,
        alpha=0.001,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        hidden_size=64,
        batch_size=32,
        buffer_size=10000,
        target_update_freq=100
    )
    dqn_rewards, dqn_steps, dqn_success, dqn_q_changes = train_agent_moving_goal(
        env, dqn_agent, episodes, grid_size, "DQN", verbose=True
    )
    
    env.close()
    
    # Summary Statistics
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Last 500 Episodes)")
    print("=" * 80)
    
    agents_data = [
        ("Tabular Q-Learning", tabular_agent, tab_rewards, tab_steps, tab_success, tab_q_changes),
        ("Linear FA (One-Hot)", linear_agent, lin_rewards, lin_steps, lin_success, lin_q_changes),
        ("DQN", dqn_agent, dqn_rewards, dqn_steps, dqn_success, dqn_q_changes)
    ]
    
    for name, agent, rewards, steps, success, q_changes in agents_data:
        final_reward = np.mean(rewards[-500:])
        final_steps = np.mean(steps[-500:])
        final_success = np.mean(success[-500:]) * 100
        final_q_change = np.mean(q_changes[-500:])
        
        print(f"\n{name}:")
        print(f"  Average Return:    {final_reward:.3f}")
        print(f"  Average Steps:     {final_steps:.2f}")
        print(f"  Success Rate:      {final_success:.2f}%")
        print(f"  Avg Q-Change:      {final_q_change:.5f}")
        if hasattr(agent, 'q_table'):
            print(f"  Q-Table Size:      {len(agent.q_table)} states")
        elif hasattr(agent, 'weights'):
            print(f"  Parameters:        {agent.weights.size}")
        elif hasattr(agent, 'w1'):
            params = agent.w1.size + agent.b1.size + agent.w2.size + agent.b2.size
            print(f"  Parameters:        {params}")
    
    # Generate Plots
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Function Approximation vs Tabular Q-Learning (Moving Goals, Goal-Blind Agents)', 
                 fontsize=14, fontweight='bold')
    
    window = 100
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = ['Tabular Q-Learning', 'Linear FA', 'DQN']
    
    all_rewards = [tab_rewards, lin_rewards, dqn_rewards]
    all_steps = [tab_steps, lin_steps, dqn_steps]
    all_success = [tab_success, lin_success, dqn_success]
    all_q_changes = [tab_q_changes, lin_q_changes, dqn_q_changes]
    
    # Plot 1: Average Return
    for i, (rewards, label, color) in enumerate(zip(all_rewards, labels, colors)):
        smooth = moving_average(rewards, window)
        axes[0, 0].plot(range(len(smooth)), smooth, 
                       label=label, linewidth=2, color=color)
    axes[0, 0].set_title('Average Return per Episode', fontweight='bold', fontsize=11)
    axes[0, 0].set_xlabel('Episode', fontsize=10)
    axes[0, 0].set_ylabel('Return (smoothed)', fontsize=10)
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    for i, (success, label, color) in enumerate(zip(all_success, labels, colors)):
        smooth = moving_average(success, window)
        axes[0, 1].plot(range(len(smooth)), smooth, 
                       label=label, linewidth=2, color=color)
    axes[0, 1].set_title('Success Rate', fontweight='bold', fontsize=11)
    axes[0, 1].set_xlabel('Episode', fontsize=10)
    axes[0, 1].set_ylabel('Success Rate (smoothed)', fontsize=10)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Average Steps
    for i, (steps, label, color) in enumerate(zip(all_steps, labels, colors)):
        smooth = moving_average(steps, window)
        axes[1, 0].plot(range(len(smooth)), smooth, 
                       label=label, linewidth=2, color=color)
    axes[1, 0].set_title('Average Steps per Episode', fontweight='bold', fontsize=11)
    axes[1, 0].set_xlabel('Episode', fontsize=10)
    axes[1, 0].set_ylabel('Steps (smoothed)', fontsize=10)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average Q-Value Change
    for i, (q_changes, label, color) in enumerate(zip(all_q_changes, labels, colors)):
        smooth = moving_average(q_changes, window)
        axes[1, 1].plot(range(len(smooth)), smooth, 
                       label=label, linewidth=2, color=color)
    axes[1, 1].set_title('Average Q-Value Change', fontweight='bold', fontsize=11)
    axes[1, 1].set_xlabel('Episode', fontsize=10)
    axes[1, 1].set_ylabel('Avg |ΔQ| (smoothed)', fontsize=10)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/function_approximation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: plots/function_approximation_comparison.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("All agents are 'goal-blind' - they only observe their position.")
    print("With randomly changing goals, no agent can learn a consistent policy.")
    print("Function approximation may generalize differently but faces same limitation.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compare Function Approximation methods vs Tabular Q-Learning'
    )
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes (default: 5000)')
    parser.add_argument('--grid_size', type=int, default=7,
                        help='Size of the grid (default: 7)')
    
    args = parser.parse_args()
    
    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    run_comparison(episodes=args.episodes, grid_size=args.grid_size)