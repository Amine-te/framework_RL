"""
Deep Q-Learning for GridWorld with Moving Goal
=======================================================

Agent now receives relative position (dx, dy) where:
  dx = goal_x - agent_x
  dy = goal_y - agent_y

This allows the network to learn goal-directed navigation that generalizes
to any goal position.

COMPARISON: Also trains with absolute state (x_agent, y_agent, x_goal, y_goal)
to compare both approaches. Goal changes EVERY episode.

Usage:
------
python deep_q_learning_moving_goal.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from src.gridworld import GridWorldEnv
from scripts.experiments.deep_q_learning_fixed_goal import (
    NaiveDQNAgent, 
    TargetDQNAgent
)
from scripts.utils.helpers import moving_average

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_relative_state(agent_pos, goal_pos):
    """
    Convert absolute positions to relative state.
    Returns (dx, dy) where dx = goal_x - agent_x, dy = goal_y - agent_y
    """
    dx = goal_pos[1] - agent_pos[1]  # goal_col - agent_col
    dy = goal_pos[0] - agent_pos[0]  # goal_row - agent_row
    return [dx, dy]


def get_absolute_state(agent_pos, goal_pos):
    """
    Return absolute state representation.
    Returns (x_agent, y_agent, x_goal, y_goal)
    """
    return [agent_pos[1], agent_pos[0], goal_pos[1], goal_pos[0]]


def get_random_goal(grid_size, avoid_positions):
    """Generate random goal position avoiding specified positions"""
    valid_goals = []
    for row in range(grid_size):
        for col in range(grid_size):
            pos = [row, col]
            if pos not in avoid_positions:
                valid_goals.append(pos)
    return random.choice(valid_goals)


def train_agent_moving_goal(agent, env, episodes=1000, goal_change_freq=1, 
                            grid_size=7, obstacles=None, verbose=True, state_type='relative'):
    """
    Train an agent with moving goals using specified state representation.
    
    Args:
        state_type: 'relative' for (dx, dy) or 'absolute' for (x_agent, y_agent, x_goal, y_goal)
    """
    if obstacles is None:
        obstacles = []
    
    rewards_history = []
    steps_history = []
    success_history = []
    loss_history = []
    goal_history = []
    
    for episode in range(episodes):
        # Change goal every N episodes
        if episode % goal_change_freq == 0:
            old_goal = env.goals[0] if env.goals else None
            goal = get_random_goal(grid_size, avoid_positions=obstacles + [[0, 0]])
            env.goals = [goal]
            if episode > 0 and verbose and goal_change_freq > 1:
                print(f"\n  → Goal moved from {old_goal} to {goal}")
        
        # Reset environment
        agent_pos, _ = env.reset(seed=SEED + episode)
        
        # Convert to appropriate state representation
        if state_type == 'relative':
            state = get_relative_state(agent_pos, env.goals[0])
        else:  # absolute
            state = get_absolute_state(agent_pos, env.goals[0])
        
        total_reward = 0
        done = False
        episode_loss = []
        
        # Track current goal for this episode
        goal_history.append(list(env.goals[0]))
        
        while not done:
            action = agent.choose_action(state, training=True)
            next_agent_pos, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Convert next position to appropriate state
            if state_type == 'relative':
                next_state = get_relative_state(next_agent_pos, env.goals[0])
            else:  # absolute
                next_state = get_absolute_state(next_agent_pos, env.goals[0])
            
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
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            avg_steps = np.mean(steps_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            avg_loss = np.mean(loss_history[-100:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Reward: {avg_reward:.2f} | Steps: {avg_steps:.1f} | "
                  f"Success: {success_rate:.1f}% | Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return rewards_history, steps_history, success_history, loss_history, goal_history


def plot_training_results(metrics):
    """Plot training results comparing both state representations"""
    metrics_relative, metrics_absolute = metrics
    rewards_rel, steps_rel, success_rel, loss_rel, goals_rel = metrics_relative
    rewards_abs, steps_abs, success_abs, loss_abs, goals_abs = metrics_absolute
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Deep Q-Learning: Moving Goal - Absolute vs Relative State Comparison\n(Goal changes every episode)', 
                 fontsize=16, fontweight='bold')
    
    # Rewards
    ax = axes[0, 0]
    ax.plot(moving_average(rewards_rel), linewidth=2, alpha=0.8, color='#2E86AB', label='Relative (dx, dy)')
    ax.plot(moving_average(rewards_abs), linewidth=2, alpha=0.8, color='#F18F01', label='Absolute (x_a, y_a, x_g, y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward (smoothed)')
    ax.set_title('Cumulative Reward per Episode', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Steps
    ax = axes[0, 1]
    ax.plot(moving_average(steps_rel), linewidth=2, alpha=0.8, color='#2E86AB', label='Relative (dx, dy)')
    ax.plot(moving_average(steps_abs), linewidth=2, alpha=0.8, color='#F18F01', label='Absolute (x_a, y_a, x_g, y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps (smoothed)')
    ax.set_title('Steps per Episode', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Success Rate
    ax = axes[1, 0]
    ax.plot(moving_average(success_rel, window=50), linewidth=2, alpha=0.8, color='#2E86AB', label='Relative (dx, dy)')
    ax.plot(moving_average(success_abs, window=50), linewidth=2, alpha=0.8, color='#F18F01', label='Absolute (x_a, y_a, x_g, y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (smoothed)')
    ax.set_title('Success Rate', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss
    ax = axes[1, 1]
    loss_rel_clipped = [min(l, 1000) for l in loss_rel]
    loss_abs_clipped = [min(l, 1000) for l in loss_abs]
    ax.plot(moving_average(loss_rel_clipped), linewidth=2, alpha=0.8, color='#2E86AB', label='Relative (dx, dy)')
    ax.plot(moving_average(loss_abs_clipped), linewidth=2, alpha=0.8, color='#F18F01', label='Absolute (x_a, y_a, x_g, y_g)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss (smoothed, clipped at 1000)')
    ax.set_title('Training Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = 'dqn_moving_goal_ep1000_grid7_state_comparison.png'
    plt.savefig(str(results_dir / filename), dpi=300, bbox_inches='tight')
    print(f"\nSaved: {results_dir / filename}")
    plt.close()


def test_agent(agent, env, num_episodes=10, render=False, state_type='relative'):
    """Test trained agent with random goals"""
    results = []
    
    for ep in range(num_episodes):
        # Random goal for each test episode
        goal = get_random_goal(env.grid_size, 
                              avoid_positions=env.obstacles + [[0, 0]])
        env.goals = [goal]
        
        agent_pos, _ = env.reset(seed=SEED + 1000 + ep)
        if state_type == 'relative':
            state = get_relative_state(agent_pos, env.goals[0])
        else:
            state = get_absolute_state(agent_pos, env.goals[0])
        
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
            next_agent_pos, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if state_type == 'relative':
                state = get_relative_state(next_agent_pos, env.goals[0])
            else:
                state = get_absolute_state(next_agent_pos, env.goals[0])
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
            'success': terminated,
            'goal': list(goal)
        })
    
    if render:
        try:
            plt.pause(1.0)
            env.close()
        except:
            pass
    
    return results


def main():
    print("=" * 70)
    print("DEEP Q-LEARNING WITH MOVING GOAL - COMPARISON")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print("Goal changes EVERY episode")
    print("\nComparing two state representations:")
    print("  1. Relative: (dx, dy) where dx = goal_x - agent_x, dy = goal_y - agent_y")
    print("  2. Absolute: (x_agent, y_agent, x_goal, y_goal)")
    print()
    
    import os
    os.makedirs('plots', exist_ok=True)
    
    obstacles = [[2, 2], [2, 3], [2, 4], [4, 4], [4, 5]]
    grid_size = 7
    
    # Create environment
    env = GridWorldEnv(
        grid_size=grid_size,
        goals=[[6, 6]],  # Initial goal (will be changed every episode)
        start_pos=[0, 0],
        obstacles=obstacles,
        reward_goal=10.0,
        reward_step=-0.1,
        max_steps=75
    )
    
    # Train with RELATIVE state representation
    print("\n" + "=" * 70)
    print("TRAINING WITH RELATIVE STATE (dx, dy)")
    print("=" * 70)
    agent_relative = TargetDQNAgent(
        state_size=2,  # (dx, dy)
        action_size=4,
        lr=0.0005,
        gamma=0.99,
        target_update_freq=100
    )
    metrics_relative = train_agent_moving_goal(
        agent_relative, env, episodes=1000, goal_change_freq=1,
        grid_size=grid_size, obstacles=obstacles, state_type='relative'
    )
    
    # Train with ABSOLUTE state representation
    print("\n" + "=" * 70)
    print("TRAINING WITH ABSOLUTE STATE (x_agent, y_agent, x_goal, y_goal)")
    print("=" * 70)
    agent_absolute = TargetDQNAgent(
        state_size=4,  # (x_agent, y_agent, x_goal, y_goal)
        action_size=4,
        lr=0.0005,
        gamma=0.99,
        target_update_freq=100
    )
    metrics_absolute = train_agent_moving_goal(
        agent_absolute, env, episodes=1000, goal_change_freq=1,
        grid_size=grid_size, obstacles=obstacles, state_type='absolute'
    )
    
    # Plot comparison
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)
    plot_training_results((metrics_relative, metrics_absolute))
    
    # Test both agents
    print("\n" + "=" * 70)
    print("TESTING BOTH AGENTS (10 random goals each)")
    print("=" * 70)
    
    print("\n--- RELATIVE STATE AGENT ---")
    test_results_rel = test_agent(agent_relative, env, num_episodes=10, render=False, state_type='relative')
    success_rate_rel = sum(r['success'] for r in test_results_rel) / len(test_results_rel) * 100
    avg_steps_rel = np.mean([r['steps'] for r in test_results_rel])
    avg_reward_rel = np.mean([r['reward'] for r in test_results_rel])
    
    print(f"Success Rate: {success_rate_rel:.1f}%")
    print(f"Avg Steps: {avg_steps_rel:.1f}")
    print(f"Avg Reward: {avg_reward_rel:.2f}")
    
    print("\n--- ABSOLUTE STATE AGENT ---")
    test_results_abs = test_agent(agent_absolute, env, num_episodes=10, render=False, state_type='absolute')
    success_rate_abs = sum(r['success'] for r in test_results_abs) / len(test_results_abs) * 100
    avg_steps_abs = np.mean([r['steps'] for r in test_results_abs])
    avg_reward_abs = np.mean([r['reward'] for r in test_results_abs])
    
    print(f"Success Rate: {success_rate_abs:.1f}%")
    print(f"Avg Steps: {avg_steps_abs:.1f}")
    print(f"Avg Reward: {avg_reward_abs:.2f}")
    
    # Visual demo with better performing agent
    print("\n" + "=" * 70)
    print("VISUAL DEMONSTRATION")
    print("=" * 70)
    if success_rate_rel >= success_rate_abs:
        print("Showing RELATIVE state agent (better performance)...")
        test_agent(agent_relative, env, num_episodes=1, render=True, state_type='relative')
    else:
        print("Showing ABSOLUTE state agent (better performance)...")
        test_agent(agent_absolute, env, num_episodes=1, render=True, state_type='absolute')
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nComparison Summary:")
    print(f"Relative State - Success: {success_rate_rel:.1f}%, Steps: {avg_steps_rel:.1f}")
    print(f"Absolute State - Success: {success_rate_abs:.1f}%, Steps: {avg_steps_abs:.1f}")
    
    plt.close('all')


if __name__ == "__main__":
    main()