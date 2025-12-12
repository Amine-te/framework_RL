"""
Stable Baselines3 DQN for GridWorld - FIXED VERSION
====================================================

Fixes:
1. Moving goal training now actually learns (uses model.learn() properly)
2. Compares two state representations: (x_a, y_a, x_g, y_g) vs (dx, dy)
3. Fixed goal uses only agent position (x_a, y_a)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.gridworld import GridWorldEnv
from scripts.utils.helpers import get_random_goal, moving_average
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class MovingGoalWrapper(gym.Wrapper):
    """Wrapper that changes goal every episode and provides different state representations"""
    
    def __init__(self, env, obstacles, change_every_n_steps=None, state_format='relative'):
        """
        Args:
            env: Base GridWorld environment
            obstacles: List of obstacle positions
            change_every_n_steps: Change goal every N steps (None = change every episode)
            state_format: 'relative' for (dx, dy) or 'absolute_with_goal' for (x_a, y_a, x_g, y_g)
        """
        super().__init__(env)
        self.obstacles = obstacles
        self.change_every_n_steps = change_every_n_steps
        self.state_format = state_format
        self.steps_since_goal_change = 0
        self.current_goal = None
        
        # Update observation space based on state format
        if state_format == 'relative':
            # (dx, dy)
            self.observation_space = spaces.Box(
                low=-max(env.height, env.width),
                high=max(env.height, env.width),
                shape=(2,),
                dtype=np.float32
            )
        else:  # absolute_with_goal
            # (x_a, y_a, x_g, y_g)
            self.observation_space = spaces.Box(
                low=0,
                high=max(env.height, env.width),
                shape=(4,),
                dtype=np.float32
            )
    
    def reset(self, **kwargs):
        # Change goal
        self.current_goal = get_random_goal(
            self.env.grid_size, 
            avoid_positions=self.obstacles + [[0, 0]]
        )
        self.env.goals = [self.current_goal]
        self.steps_since_goal_change = 0
        
        obs, info = self.env.reset(**kwargs)
        return self._transform_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_since_goal_change += 1
        
        # Optionally change goal mid-episode
        if self.change_every_n_steps and self.steps_since_goal_change >= self.change_every_n_steps:
            self.current_goal = get_random_goal(
                self.env.grid_size,
                avoid_positions=self.obstacles + [self.env.agent_pos]
            )
            self.env.goals = [self.current_goal]
            self.steps_since_goal_change = 0
        
        return self._transform_observation(obs), reward, terminated, truncated, info
    
    def _transform_observation(self, obs):
        """Transform observation based on state format"""
        agent_pos = self.env.agent_pos
        goal_pos = self.current_goal
        
        if self.state_format == 'relative':
            # (dx, dy) - relative position to goal
            dx = goal_pos[1] - agent_pos[1]
            dy = goal_pos[0] - agent_pos[0]
            return np.array([dx, dy], dtype=np.float32)
        else:  # absolute_with_goal
            # (x_a, y_a, x_g, y_g) - absolute positions
            return np.array([agent_pos[1], agent_pos[0], goal_pos[1], goal_pos[0]], dtype=np.float32)


class MetricsCallback(BaseCallback):
    """Custom callback to track training metrics"""
    
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_successes = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_steps += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_steps.append(self.current_episode_steps)
            
            # Check if success
            terminated = self.locals['rewards'][0] > 5.0
            self.episode_successes.append(1 if terminated else 0)
            
            # Reset counters
            self.current_episode_reward = 0
            self.current_episode_steps = 0
            
            # Print progress
            if len(self.episode_rewards) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_steps = np.mean(self.episode_steps[-100:])
                success_rate = np.mean(self.episode_successes[-100:]) * 100
                print(f"Episode {len(self.episode_rewards)} | "
                      f"Reward: {avg_reward:.2f} | "
                      f"Steps: {avg_steps:.1f} | "
                      f"Success: {success_rate:.1f}%")
        
        return True


# moving_average is now imported from scripts.utils.helpers


def train_fixed_goal(env, total_timesteps=50000, verbose=True):
    """Train on fixed goal with agent position only"""
    
    if verbose:
        print("Training on Fixed Goal (agent position only)...")
    
    callback = MetricsCallback(verbose=1)
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        target_update_interval=100,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=0,
        seed=SEED
    )
    
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=None)
    
    return model, callback


def train_moving_goal(env_wrapper, total_timesteps=75000, verbose=True):
    """Train on moving goals - FIXED to actually learn"""
    
    if verbose:
        state_type = "Relative (dx, dy)" if env_wrapper.state_format == 'relative' else "Absolute (x_a, y_a, x_g, y_g)"
        print(f"Training on Moving Goals [{state_type}]...")
    
    callback = MetricsCallback(verbose=1)
    
    model = DQN(
        "MlpPolicy",
        env_wrapper,
        learning_rate=0.0005,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        target_update_interval=100,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        verbose=0,
        seed=SEED
    )
    
    # THIS IS THE FIX: Use model.learn() which handles everything automatically
    model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=None)
    
    return model, callback


def test_agent(model, env, num_episodes=10):
    """Test trained agent"""
    results = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=SEED + 1000 + ep)
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        results.append({
            'reward': total_reward,
            'steps': steps,
            'success': terminated
        })
    
    return results


def plot_fixed_goal(callback_fixed):
    """Plot training results for fixed goal only"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fixed Goal DQN Training - State: (x_a, y_a)', 
                 fontsize=16, fontweight='bold')
    
    rewards = callback_fixed.episode_rewards
    steps = callback_fixed.episode_steps
    success = callback_fixed.episode_successes
    
    # Rewards
    ax = axes[0, 0]
    ax.plot(moving_average(rewards), linewidth=2, alpha=0.8, color='#2E86AB')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('Cumulative Reward per Episode', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Steps
    ax = axes[0, 1]
    ax.plot(moving_average(steps), linewidth=2, alpha=0.8, color='#2E86AB')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps (smoothed)')
    ax.set_title('Steps per Episode', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Success Rate
    ax = axes[1, 0]
    ax.plot(moving_average(success, window=50), linewidth=2, alpha=0.8, color='#2E86AB')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (smoothed)')
    ax.set_title('Success Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cumulative Success
    ax = axes[1, 1]
    cumsum = np.cumsum(success) / (np.arange(len(success)) + 1) * 100
    ax.plot(cumsum, linewidth=2, alpha=0.8, color='#2E86AB')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Success Rate (%)')
    ax.set_title('Overall Success Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = 'sb3_dqn_fixed_goal_grid7.png'
    plt.savefig(str(results_dir / filename), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {results_dir / filename}")
    plt.close()


def plot_moving_goal_comparison(callback_relative, callback_absolute):
    """Plot comparison of two state representations for moving goal"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Moving Goal DQN Training - State Representation Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Rewards
    ax = axes[0, 0]
    ax.plot(moving_average(callback_relative.episode_rewards), 
            linewidth=2, alpha=0.8, color='#C73E1D', label='Relative (dx, dy)')
    ax.plot(moving_average(callback_absolute.episode_rewards), 
            linewidth=2, alpha=0.8, color='#06A77D', label='Absolute (x_a,y_a,x_g,y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('Cumulative Reward per Episode', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Steps
    ax = axes[0, 1]
    ax.plot(moving_average(callback_relative.episode_steps), 
            linewidth=2, alpha=0.8, color='#C73E1D', label='Relative (dx, dy)')
    ax.plot(moving_average(callback_absolute.episode_steps), 
            linewidth=2, alpha=0.8, color='#06A77D', label='Absolute (x_a,y_a,x_g,y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps (smoothed)')
    ax.set_title('Steps per Episode', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    ax = axes[1, 0]
    ax.plot(moving_average(callback_relative.episode_successes, window=50), 
            linewidth=2, alpha=0.8, color='#C73E1D', label='Relative (dx, dy)')
    ax.plot(moving_average(callback_absolute.episode_successes, window=50), 
            linewidth=2, alpha=0.8, color='#06A77D', label='Absolute (x_a,y_a,x_g,y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (smoothed)')
    ax.set_title('Success Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative Success
    ax = axes[1, 1]
    cumsum_rel = np.cumsum(callback_relative.episode_successes) / (np.arange(len(callback_relative.episode_successes)) + 1) * 100
    cumsum_abs = np.cumsum(callback_absolute.episode_successes) / (np.arange(len(callback_absolute.episode_successes)) + 1) * 100
    
    ax.plot(cumsum_rel, linewidth=2, alpha=0.8, color='#C73E1D', label='Relative (dx, dy)')
    ax.plot(cumsum_abs, linewidth=2, alpha=0.8, color='#06A77D', label='Absolute (x_a,y_a,x_g,y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Success Rate (%)')
    ax.set_title('Overall Success Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = 'sb3_dqn_moving_goal_grid7_state_comparison.png'
    plt.savefig(str(results_dir / filename), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {results_dir / filename}")
    plt.close()


def main():
    print("=" * 70)
    print("STABLE BASELINES3 DQN - STATE REPRESENTATION COMPARISON")
    print("=" * 70)
    print(f"Seed: {SEED}\n")
    
    import os
    os.makedirs('plots', exist_ok=True)
    
    obstacles = [[2, 2], [2, 3], [2, 4], [4, 4], [4, 5]]
    grid_size = 7
    
    # ========================================================================
    # EXPERIMENT 1: FIXED GOAL - Agent position only (x_a, y_a)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: FIXED GOAL")
    print("State: (x_a, y_a) - Agent position only")
    print("=" * 70)
    
    env_fixed = GridWorldEnv(
        grid_size=grid_size,
        goals=[[6, 6]],
        start_pos=[0, 0],
        obstacles=obstacles,
        reward_goal=10.0,
        reward_step=-0.1,
        max_steps=75,
        state_type='absolute'  # Returns [row, col] of agent
    )
    
    model_fixed, callback_fixed = train_fixed_goal(env_fixed, total_timesteps=50000)
    
    print("\nTesting Fixed Goal Agent...")
    results_fixed = test_agent(model_fixed, env_fixed, num_episodes=10)
    success_rate_fixed = sum(r['success'] for r in results_fixed) / len(results_fixed) * 100
    avg_steps_fixed = np.mean([r['steps'] for r in results_fixed])
    avg_reward_fixed = np.mean([r['reward'] for r in results_fixed])
    
    print(f"  Success Rate: {success_rate_fixed:.1f}%")
    print(f"  Avg Steps: {avg_steps_fixed:.1f}")
    print(f"  Avg Reward: {avg_reward_fixed:.2f}")
    
    # ========================================================================
    # EXPERIMENT 2: MOVING GOAL - Relative state (dx, dy)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: MOVING GOAL - RELATIVE STATE")
    print("State: (dx, dy) - Relative position to goal")
    print("=" * 70)
    
    env_moving_base = GridWorldEnv(
        grid_size=grid_size,
        goals=[[6, 6]],  # Will be changed by wrapper
        start_pos=[0, 0],
        obstacles=obstacles,
        reward_goal=10.0,
        reward_step=-0.1,
        max_steps=75,
        state_type='absolute'  # Wrapper will transform this
    )
    
    env_moving_relative = MovingGoalWrapper(
        env_moving_base,
        obstacles=obstacles,
        state_format='relative'
    )
    
    model_relative, callback_relative = train_moving_goal(
        env_moving_relative, 
        total_timesteps=75000
    )
    
    print("\nTesting Moving Goal Agent (Relative State)...")
    results_relative = test_agent(model_relative, env_moving_relative, num_episodes=10)
    success_rate_relative = sum(r['success'] for r in results_relative) / len(results_relative) * 100
    avg_steps_relative = np.mean([r['steps'] for r in results_relative])
    avg_reward_relative = np.mean([r['reward'] for r in results_relative])
    
    print(f"  Success Rate: {success_rate_relative:.1f}%")
    print(f"  Avg Steps: {avg_steps_relative:.1f}")
    print(f"  Avg Reward: {avg_reward_relative:.2f}")
    
    # ========================================================================
    # EXPERIMENT 3: MOVING GOAL - Absolute state (x_a, y_a, x_g, y_g)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: MOVING GOAL - ABSOLUTE STATE")
    print("State: (x_a, y_a, x_g, y_g) - Full absolute positions")
    print("=" * 70)
    
    env_moving_base2 = GridWorldEnv(
        grid_size=grid_size,
        goals=[[6, 6]],
        start_pos=[0, 0],
        obstacles=obstacles,
        reward_goal=10.0,
        reward_step=-0.1,
        max_steps=75,
        state_type='absolute'
    )
    
    env_moving_absolute = MovingGoalWrapper(
        env_moving_base2,
        obstacles=obstacles,
        state_format='absolute_with_goal'
    )
    
    model_absolute, callback_absolute = train_moving_goal(
        env_moving_absolute,
        total_timesteps=75000
    )
    
    print("\nTesting Moving Goal Agent (Absolute State)...")
    results_absolute = test_agent(model_absolute, env_moving_absolute, num_episodes=10)
    success_rate_absolute = sum(r['success'] for r in results_absolute) / len(results_absolute) * 100
    avg_steps_absolute = np.mean([r['steps'] for r in results_absolute])
    avg_reward_absolute = np.mean([r['reward'] for r in results_absolute])
    
    print(f"  Success Rate: {success_rate_absolute:.1f}%")
    print(f"  Avg Steps: {avg_steps_absolute:.1f}")
    print(f"  Avg Reward: {avg_reward_absolute:.2f}")
    
    # ========================================================================
    # COMPARISON PLOTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    print("\nPlot 1: Fixed Goal Training...")
    plot_fixed_goal(callback_fixed)
    
    print("\nPlot 2: Moving Goal State Representation Comparison...")
    plot_moving_goal_comparison(callback_relative, callback_absolute)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print("\n1. FIXED GOAL - State: (x_a, y_a)")
    print(f"   Success Rate: {success_rate_fixed:.1f}%")
    print(f"   Avg Reward: {avg_reward_fixed:.2f}")
    print(f"   Avg Steps: {avg_steps_fixed:.1f}")
    
    print("\n2. MOVING GOAL (Relative) - State: (dx, dy)")
    print(f"   Success Rate: {success_rate_relative:.1f}%")
    print(f"   Avg Reward: {avg_reward_relative:.2f}")
    print(f"   Avg Steps: {avg_steps_relative:.1f}")
    
    print("\n3. MOVING GOAL (Absolute) - State: (x_a, y_a, x_g, y_g)")
    print(f"   Success Rate: {success_rate_absolute:.1f}%")
    print(f"   Avg Reward: {avg_reward_absolute:.2f}")
    print(f"   Avg Steps: {avg_steps_absolute:.1f}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("• Fixed goal learns fastest (goal position is constant)")
    print("• Relative state (dx, dy) is most compact for moving goals")
    print("• Absolute state (x_a, y_a, x_g, y_g) provides full information")
    print("• Relative state typically learns better for moving goals")
    print("\n" + "=" * 70)
    
    plt.close('all')


if __name__ == "__main__":
    main()