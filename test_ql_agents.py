"""
Test Trained Q-Learning Agents and Generate GIF Visualizations
================================================================

This script loads trained Q-Learning agents and creates GIF animations
showing their behavior on the grid world.

Requirements:
    pip install pillow

Usage:
------
python test_ql_agents.py
python test_ql_agents.py --agents g5_a0.3_gm0.99 g7_a0.5_gm0.99  # Specific agents
python test_ql_agents.py --all  # Test all available agents
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import argparse
from pathlib import Path
from PIL import Image
import io
import random


class GridWorldEnv:
    """GridWorld environment (matches q_learning.py)"""
    
    def __init__(self, grid_size=5, goal_pos=None, start_pos=None, 
                 obstacles=None, reward_goal=10.0, reward_step=-0.1, 
                 reward_obstacle=-5.0, max_steps=100):
        
        self.grid_size = grid_size
        self.goal_pos = goal_pos if goal_pos else [grid_size-1, grid_size-1]
        self.start_pos = start_pos
        self.obstacles = obstacles if obstacles else []
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_obstacle = reward_obstacle
        self.max_steps = max_steps
        self.num_actions = 4
        
        self.agent_pos = None
        self.episode_steps = 0
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        if self.start_pos is not None:
            self.agent_pos = list(self.start_pos)
        else:
            while True:
                row = np.random.randint(0, self.grid_size)
                col = np.random.randint(0, self.grid_size)
                self.agent_pos = [row, col]
                
                if (self.agent_pos != self.goal_pos and 
                    self.agent_pos not in self.obstacles):
                    break
        
        self.episode_steps = 0
        observation = list(self.agent_pos)
        info = {
            'distance_to_goal': self._calculate_distance(),
            'episode_steps': self.episode_steps
        }
        
        return observation, info
    
    def step(self, action):
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action {action}")
        
        new_row = self.agent_pos[0]
        new_col = self.agent_pos[1]
        
        if action == 0:      # UP
            new_row -= 1
        elif action == 1:    # RIGHT
            new_col += 1
        elif action == 2:    # DOWN
            new_row += 1
        elif action == 3:    # LEFT
            new_col -= 1
        
        # Check if new position is valid and not an obstacle
        if self._is_valid_position(new_row, new_col):
            if [new_row, new_col] not in self.obstacles:
                self.agent_pos = [new_row, new_col]
            else:
                reward = self.reward_obstacle
                self.episode_steps += 1
                observation = list(self.agent_pos)
                info = {
                    'distance_to_goal': self._calculate_distance(),
                    'episode_steps': self.episode_steps,
                    'hit_obstacle': True
                }
                return observation, reward, False, False, info
        
        self.episode_steps += 1
        
        terminated = (self.agent_pos == self.goal_pos)
        truncated = (self.episode_steps >= self.max_steps)
        
        if terminated:
            reward = self.reward_goal
        else:
            reward = self.reward_step
        
        observation = list(self.agent_pos)
        info = {
            'distance_to_goal': self._calculate_distance(),
            'episode_steps': self.episode_steps,
            'hit_obstacle': False
        }
        
        return observation, reward, terminated, truncated, info
    
    def _is_valid_position(self, row, col):
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size
    
    def _calculate_distance(self):
        return abs(self.agent_pos[0] - self.goal_pos[0]) + \
               abs(self.agent_pos[1] - self.goal_pos[1])


class QLearningAgent:
    """Q-Learning Agent (simplified for testing)"""
    
    def __init__(self, env, Q_table, gamma, alpha):
        self.env = env
        self.Q = Q_table
        self.gamma = gamma
        self.alpha = alpha
    
    def get_action(self, observation):
        """Get greedy action (no exploration during testing)"""
        state = tuple(observation)
        return np.argmax(self.Q[state])
    
    @classmethod
    def load(cls, filepath, env):
        """Load agent from file"""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        from collections import defaultdict
        Q_table = defaultdict(lambda: np.zeros(4), agent_data['Q'])
        
        return cls(env, Q_table, agent_data['gamma'], agent_data['alpha'])


def render_frame(env, episode_step, max_steps, total_reward):
    """Render a single frame of the environment"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid lines
    for i in range(env.grid_size + 1):
        ax.axhline(i, color='black', linewidth=1.5)
        ax.axvline(i, color='black', linewidth=1.5)
    
    # Draw obstacles (gray)
    for obs in env.obstacles:
        x = obs[1]
        y = env.grid_size - obs[0] - 1
        rect = Rectangle((x, y), 1, 1, facecolor='gray', 
                       edgecolor='black', linewidth=2)
        ax.add_patch(rect)
    
    # Draw goal (red)
    x = env.goal_pos[1]
    y = env.grid_size - env.goal_pos[0] - 1
    rect = Rectangle((x, y), 1, 1, facecolor='red', 
                   edgecolor='darkred', linewidth=3)
    ax.add_patch(rect)
    
    # Draw agent (green)
    if env.agent_pos is not None:
        x = env.agent_pos[1]
        y = env.grid_size - env.agent_pos[0] - 1
        rect = Rectangle((x, y), 1, 1, facecolor='green', 
                       edgecolor='darkgreen', linewidth=3)
        ax.add_patch(rect)
    
    # Configure plot
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.set_title(f'Step {episode_step}/{max_steps} | Total Reward: {total_reward:.1f}', 
                fontsize=16, fontweight='bold')
    
    # Add grid labels
    ax.set_xticks(range(env.grid_size + 1))
    ax.set_yticks(range(env.grid_size + 1))
    ax.tick_params(labelbottom=False, labelleft=False)
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def test_agent_and_create_gif(agent_path, output_dir, seed=42):
    """Test a single agent and create a GIF of its performance"""
    
    # Parse agent filename to get parameters
    filename = agent_path.stem  # e.g., "agent_g5_a0.3_gm0.99"
    parts = filename.replace('agent_', '').split('_')
    
    grid_size = int(parts[0][1:])  # g5 -> 5
    alpha = float(parts[1][1:])     # a0.3 -> 0.3
    gamma = float(parts[2][2:])     # gm0.99 -> 0.99
    
    # Set up obstacles (same as in q_learning.py)
    obstacles_config = {
        5: [[1, 2], [2, 2]],
        7: [[2, 3], [3, 3], [4, 3]],
        10: [[3, 5], [4, 5], [5, 5], [6, 5]]
    }
    obstacles = obstacles_config.get(grid_size, [])
    
    # Create environment
    env = GridWorldEnv(
        grid_size=grid_size,
        goal_pos=[grid_size-1, grid_size-1],
        start_pos=[0, 0],
        obstacles=obstacles,
        reward_goal=10.0,
        reward_step=-0.1,
        reward_obstacle=-5.0,
        max_steps=grid_size * 20
    )
    
    # Load agent
    print(f"Loading agent: {agent_path.name}")
    agent = QLearningAgent.load(agent_path, env)
    
    # Run episode and collect frames
    frames = []
    obs, info = env.reset(seed=seed)
    total_reward = 0
    done = False
    
    # Add initial frame
    frames.append(render_frame(env, 0, env.max_steps, total_reward))
    
    while not done:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # Add frame
        frames.append(render_frame(env, env.episode_steps, env.max_steps, total_reward))
    
    # Add a few extra frames at the end to pause
    for _ in range(5):
        frames.append(frames[-1])
    
    # Save as GIF
    output_filename = f"{filename}.gif"
    output_path = output_dir / output_filename
    
    # Determine duration based on success
    if terminated:
        duration = 300  # 300ms per frame for successful episodes
        status = "SUCCESS"
    else:
        duration = 100  # Faster for truncated episodes
        status = "TRUNCATED"
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"  Grid: {grid_size}x{grid_size} | α={alpha} | γ={gamma}")
    print(f"  Status: {status} | Steps: {env.episode_steps} | Reward: {total_reward:.2f}")
    print(f"  GIF saved: {output_path}\n")
    
    return {
        'grid_size': grid_size,
        'alpha': alpha,
        'gamma': gamma,
        'steps': env.episode_steps,
        'reward': total_reward,
        'success': terminated
    }


def main():
    parser = argparse.ArgumentParser(description='Test Q-Learning agents and create GIFs')
    parser.add_argument('--agents', nargs='+', 
                       help='Specific agents to test (e.g., g5_a0.3_gm0.99)')
    parser.add_argument('--all', action='store_true',
                       help='Test all available agents')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='gifs',
                       help='Output directory for GIFs (default: gifs)')
    
    args = parser.parse_args()
    
    # Set up directories
    agents_dir = Path("agents")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not agents_dir.exists():
        print(f"Error: 'agents/' directory not found!")
        print("Please run q_learning.py first to train agents.")
        return
    
    # Get list of agents to test
    if args.all:
        agent_files = sorted(agents_dir.glob("agent_*.pkl"))
    elif args.agents:
        agent_files = []
        for agent_spec in args.agents:
            # Allow both formats: "g5_a0.3_gm0.99" or "agent_g5_a0.3_gm0.99"
            if not agent_spec.startswith('agent_'):
                agent_spec = f'agent_{agent_spec}'
            if not agent_spec.endswith('.pkl'):
                agent_spec = f'{agent_spec}.pkl'
            
            agent_path = agents_dir / agent_spec
            if agent_path.exists():
                agent_files.append(agent_path)
            else:
                print(f"Warning: Agent file not found: {agent_path}")
    else:
        # Default: test best agents from each grid size
        default_agents = [
            "agent_g5_a0.3_gm0.99.pkl",
            "agent_g7_a0.3_gm0.99.pkl",
            "agent_g10_a0.3_gm0.99.pkl"
        ]
        agent_files = [agents_dir / name for name in default_agents if (agents_dir / name).exists()]
        
        if not agent_files:
            print("No default agents found. Available agents:")
            for f in sorted(agents_dir.glob("agent_*.pkl")):
                print(f"  - {f.name}")
            print("\nUse --all to test all agents or specify agents with --agents")
            return
    
    if not agent_files:
        print("No agents to test!")
        return
    
    print(f"\n{'='*80}")
    print(f"TESTING {len(agent_files)} AGENTS")
    print(f"{'='*80}\n")
    
    # Test each agent
    results = []
    for agent_path in agent_files:
        result = test_agent_and_create_gif(agent_path, output_dir, seed=args.seed)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Grid':<8} {'Alpha':<8} {'Gamma':<8} {'Steps':<8} {'Reward':<10} {'Status'}")
    print("-" * 70)
    
    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ TRUNCATED"
        print(f"{r['grid_size']:>4}x{r['grid_size']:<2}  "
              f"{r['alpha']:<8.2f} {r['gamma']:<8.2f} "
              f"{r['steps']:<8} {r['reward']:<10.2f} {status}")
    
    print(f"\n{len(results)} GIFs saved to '{output_dir}/' directory")
    print("\nTo view GIFs, open them in a web browser or image viewer.")


if __name__ == "__main__":
    main()