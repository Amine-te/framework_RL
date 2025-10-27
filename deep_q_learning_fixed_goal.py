"""
Deep Q-Learning for GridWorld
==============================

Two approaches:
1. Naive DQN: Uses same network for prediction and target (unstable)
2. Target DQN: Uses separate target network updated periodically (stable)

Usage:
------
python deep_q_learning.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gridworld import GridWorldEnv

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class SimpleQNetwork(nn.Module):
    """Simple fully connected network for Q-values"""
    
    def __init__(self, state_size=2, action_size=4, hidden_size=128):
        super(SimpleQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Simple experience replay buffer"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class NaiveDQNAgent:
    """
    Naive DQN: Uses same network for both prediction and target.
    Problem: "chasing its own tail" - target moves as we update the network.
    """
    
    def __init__(self, state_size=2, action_size=4, lr=0.0005, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Single network for everything
        self.network = SimpleQNetwork(state_size, action_size, hidden_size=128)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(capacity=5000)
        self.batch_size = 64
    
    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Train on a batch from memory"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # NAIVE: Use same network for target (unstable!)
        with torch.no_grad():
            next_q = self.network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class TargetDQNAgent:
    """
    Target DQN (DeepMind approach): Uses two networks.
    - Online network: Updated every step
    - Target network: Copy of online, updated periodically (stable targets)
    """
    
    def __init__(self, state_size=2, action_size=4, lr=0.0005, gamma=0.99, 
                 target_update_freq=100):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Online network (updated every step)
        self.online_network = SimpleQNetwork(state_size, action_size, hidden_size=128)
        
        # Target network (updated periodically)
        self.target_network = SimpleQNetwork(state_size, action_size, hidden_size=128)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.memory = ReplayBuffer(capacity=5000)
        self.batch_size = 64
    
    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection using online network"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.online_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Train online network, periodically update target network"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Current Q values from online network
        current_q = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # TARGET: Use target network for stable Q-value estimates
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update online network
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def create_7x7_env():
    """Create 7x7 GridWorld with obstacles"""
    return GridWorldEnv(
        grid_size=7,
        goals=[[6, 6]],
        start_pos=[0, 0],
        obstacles=[[2, 2], [2, 3], [2, 4], [4, 4], [4, 5]],
        reward_goal=10.0,
        reward_step=-0.1,
        max_steps=75
    )


def train_agent(agent, env, episodes=500, verbose=True):
    """Train an agent and return metrics"""
    rewards_history = []
    steps_history = []
    success_history = []
    loss_history = []
    
    for episode in range(episodes):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0
        done = False
        episode_loss = []
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            if loss > 0:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        steps_history.append(env.episode_steps)
        success_history.append(1 if terminated else 0)
        loss_history.append(np.mean(episode_loss) if episode_loss else 0)
        
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:])
            avg_steps = np.mean(steps_history[-50:])
            success_rate = np.mean(success_history[-50:]) * 100
            avg_loss = np.mean(loss_history[-50:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Reward: {avg_reward:.2f} | Steps: {avg_steps:.1f} | "
                  f"Success: {success_rate:.1f}% | Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history, success_history, loss_history


def moving_average(data, window=20):
    """Smooth data with moving average"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def test_agent(agent, env, num_episodes=10, render=False):
    """Test trained agent"""
    results = []
    
    for ep in range(num_episodes):
        state, _ = env.reset(seed=SEED + 1000 + ep)
        total_reward = 0
        done = False
        
        if render and ep == 0:
            try:
                env.render_init()
                env.render_update()
                plt.pause(0.3)
            except:
                print("Warning: Could not initialize rendering")
                render = False
        
        while not done:
            action = agent.choose_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            if render and ep == 0:
                try:
                    env.render_update()
                    plt.pause(0.2)
                except:
                    pass
        
        results.append({
            'reward': total_reward,
            'steps': env.episode_steps,
            'success': terminated
        })
    
    if render:
        try:
            plt.pause(1.0)
            env.close()
        except:
            pass
    
    return results


def plot_comparison(naive_metrics, target_metrics):
    """Plot training comparison between both approaches"""
    naive_rewards, naive_steps, naive_success, naive_loss = naive_metrics
    target_rewards, target_steps, target_success, target_loss = target_metrics
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Deep Q-Learning: Naive vs Target Network Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Rewards
    ax = axes[0, 0]
    ax.plot(moving_average(naive_rewards), label='Naive DQN', linewidth=2, alpha=0.8)
    ax.plot(moving_average(target_rewards), label='Target DQN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('Cumulative Reward per Episode', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Steps
    ax = axes[0, 1]
    ax.plot(moving_average(naive_steps), label='Naive DQN', linewidth=2, alpha=0.8)
    ax.plot(moving_average(target_steps), label='Target DQN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps (smoothed)')
    ax.set_title('Steps per Episode', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success Rate
    ax = axes[1, 0]
    ax.plot(moving_average(naive_success, window=50), label='Naive DQN', 
            linewidth=2, alpha=0.8)
    ax.plot(moving_average(target_success, window=50), label='Target DQN', 
            linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (smoothed)')
    ax.set_title('Success Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[1, 1]
    # Clip loss for better visualization
    naive_loss_clipped = [min(l, 1000) for l in naive_loss]
    target_loss_clipped = [min(l, 1000) for l in target_loss]
    ax.plot(moving_average(naive_loss_clipped), label='Naive DQN', linewidth=2, alpha=0.8)
    ax.plot(moving_average(target_loss_clipped), label='Target DQN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss (smoothed, clipped at 1000)')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/dqn_comparison_fixed_goal.png', dpi=300, bbox_inches='tight')
    print("\nSaved: plots/dqn_comparison_fixed_goal.png")
    plt.close()


def main():
    print("=" * 70)
    print("DEEP Q-LEARNING FOR GRIDWORLD (7x7 with obstacles)")
    print("=" * 70)
    print(f"Seed: {SEED}\n")
    
    import os
    os.makedirs('plots', exist_ok=True)
    
    env = create_7x7_env()
    
    # Train Naive DQN
    print("\n" + "=" * 70)
    print("TRAINING NAIVE DQN (unstable targets)")
    print("=" * 70)
    naive_agent = NaiveDQNAgent(state_size=2, action_size=4, lr=0.0005, gamma=0.99)
    naive_metrics = train_agent(naive_agent, env, episodes=500)
    
    # Train Target DQN
    print("\n" + "=" * 70)
    print("TRAINING TARGET DQN (stable targets)")
    print("=" * 70)
    target_agent = TargetDQNAgent(state_size=2, action_size=4, lr=0.0005, gamma=0.99,
                                   target_update_freq=100)
    target_metrics = train_agent(target_agent, env, episodes=500)
    
    # Plot comparison
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_comparison(naive_metrics, target_metrics)
    
    # Test both agents
    print("\n" + "=" * 70)
    print("TESTING TRAINED AGENTS")
    print("=" * 70)
    
    print("\nNaive DQN Test (10 episodes):")
    naive_results = test_agent(naive_agent, env, num_episodes=10, render=False)
    naive_success = sum(r['success'] for r in naive_results) / len(naive_results) * 100
    naive_avg_steps = np.mean([r['steps'] for r in naive_results])
    naive_avg_reward = np.mean([r['reward'] for r in naive_results])
    print(f"  Success Rate: {naive_success:.1f}%")
    print(f"  Avg Steps: {naive_avg_steps:.1f}")
    print(f"  Avg Reward: {naive_avg_reward:.2f}")
    
    print("\nTarget DQN Test (10 episodes):")
    target_results = test_agent(target_agent, env, num_episodes=10, render=False)
    target_success = sum(r['success'] for r in target_results) / len(target_results) * 100
    target_avg_steps = np.mean([r['steps'] for r in target_results])
    target_avg_reward = np.mean([r['reward'] for r in target_results])
    print(f"  Success Rate: {target_success:.1f}%")
    print(f"  Avg Steps: {target_avg_steps:.1f}")
    print(f"  Avg Reward: {target_avg_reward:.2f}")
    
    # Visual demo of best agent
    print("\n" + "=" * 70)
    print("VISUAL DEMONSTRATION (Best Agent)")
    print("=" * 70)
    if target_success >= naive_success:
        print("Showing Target DQN agent...")
        test_agent(target_agent, env, num_episodes=1, render=True)
    else:
        print("Showing Naive DQN agent...")
        test_agent(naive_agent, env, num_episodes=1, render=True)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  Naive DQN:  {naive_success:.1f}% success, {naive_avg_reward:.2f} avg reward")
    print(f"  Target DQN: {target_success:.1f}% success, {target_avg_reward:.2f} avg reward")
    print(f"\nThe Target DQN approach with separate networks provides more")
    print(f"stable training by preventing the 'chasing its tail' problem.")
    
    plt.close('all')


if __name__ == "__main__":
    main()