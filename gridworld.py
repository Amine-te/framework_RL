"""
Modular GridWorld Environment with Smooth Animation
A clean, parametrized framework for reinforcement learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random


class GridWorldEnv:
    """
    Customizable grid world environment.
    
    Parameters:
        grid_size (int): Size of the square grid (used if width/height not specified)
        width (int): Width of the grid (columns). If None, uses grid_size
        height (int): Height of the grid (rows). If None, uses grid_size
        goals (list): List of goal positions [[row, col], ...] or single [row, col]
        start_pos (list): Starting position [row, col] (None for random)
        obstacles (list): List of obstacle positions [[row, col], ...]
        reward_goal (float): Reward for reaching a goal
        reward_step (float): Reward per step (usually negative)
        max_steps (int): Maximum steps per episode
    """
    
    def __init__(self, grid_size=5, width=None, height=None, goals=None, start_pos=None, 
                 obstacles=None, reward_goal=10.0, reward_step=-0.1, max_steps=50):
        
        # Handle rectangular grids: if width/height not specified, use grid_size
        self.width = width if width is not None else grid_size
        self.height = height if height is not None else grid_size
        
        # Keep grid_size for backward compatibility (use height as primary dimension)
        self.grid_size = self.height
        
        # Handle goals: convert single goal to list format
        if goals is None:
            self.goals = [[self.height-1, self.width-1]]
        elif isinstance(goals[0], int):
            self.goals = [goals]  # Single goal [row, col] -> [[row, col]]
        else:
            self.goals = goals
        
        # Handle obstacles
        self.obstacles = obstacles if obstacles is not None else []
        
        self.start_pos = start_pos
        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.max_steps = max_steps
        self.num_actions = 4  # UP, RIGHT, DOWN, LEFT
        
        self.agent_pos = None
        self.episode_steps = 0
        
        # Rendering attributes
        self.fig = None
        self.ax = None
        self.agent_patch = None
    
    def reset(self, seed=None):
        """Start a new episode."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)  # FIX: Also seed Python's random module
        
        # Set starting position
        if self.start_pos is not None:
            self.agent_pos = list(self.start_pos)
        else:
            # Random start, avoiding goals and obstacles
            occupied = self.goals + self.obstacles
            while True:
                row = np.random.randint(0, self.height)
                col = np.random.randint(0, self.width)
                self.agent_pos = [row, col]
                if self.agent_pos not in occupied:
                    break
        
        self.episode_steps = 0
        
        observation = list(self.agent_pos)
        info = {
            'distance_to_goal': self._min_distance_to_goal(),
            'episode_steps': self.episode_steps
        }
        
        return observation, info
    
    def step(self, action):
        """Execute one action."""
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Invalid action {action}")
        
        # Calculate new position
        new_row, new_col = self.agent_pos[0], self.agent_pos[1]
        
        if action == 0:    # UP
            new_row -= 1
        elif action == 1:  # RIGHT
            new_col += 1
        elif action == 2:  # DOWN
            new_row += 1
        elif action == 3:  # LEFT
            new_col -= 1
        
        # Check if move is valid (within grid and not an obstacle)
        if self._is_valid_position(new_row, new_col) and \
           [new_row, new_col] not in self.obstacles:
            self.agent_pos = [new_row, new_col]
        
        self.episode_steps += 1
        
        # Check termination
        terminated = self.agent_pos in self.goals
        truncated = self.episode_steps >= self.max_steps
        
        # Calculate reward
        reward = self.reward_goal if terminated else self.reward_step
        
        observation = list(self.agent_pos)
        info = {
            'distance_to_goal': self._min_distance_to_goal(),
            'episode_steps': self.episode_steps
        }
        
        return observation, reward, terminated, truncated, info
    
    def render_init(self):
        """Initialize the rendering window (call once at start)."""
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        plt.ion()  # Interactive mode
        
        # Draw grid lines
        for i in range(self.height + 1):
            self.ax.axhline(i, color='black', linewidth=1)
        for i in range(self.width + 1):
            self.ax.axvline(i, color='black', linewidth=1)
        
        # Draw obstacles (gray)
        for obs in self.obstacles:
            x = obs[1]
            y = self.height - obs[0] - 1
            rect = Rectangle((x, y), 1, 1, facecolor='gray', 
                           edgecolor='black', linewidth=2)
            self.ax.add_patch(rect)
        
        # Draw goals (red)
        for goal in self.goals:
            x = goal[1]
            y = self.height - goal[0] - 1
            rect = Rectangle((x, y), 1, 1, facecolor='red', 
                           edgecolor='darkred', linewidth=3, label='Goal')
            self.ax.add_patch(rect)
        
        # Create agent patch (initially hidden)
        self.agent_patch = Rectangle((0, 0), 1, 1, facecolor='green', 
                                     edgecolor='darkgreen', linewidth=3, label='Agent')
        self.ax.add_patch(self.agent_patch)
        
        # Configure plot
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_title(f'GridWorld - Step 0/{self.max_steps}\nPress Q to quit', 
                         fontsize=14, fontweight='bold')
        
        # Remove duplicate labels in legend
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
                      bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)
    
    def render_update(self):
        """Update only the agent's position (fast)."""
        if self.fig is None or self.ax is None:
            self.render_init()
        
        # Update agent position
        if self.agent_pos is not None:
            x = self.agent_pos[1]
            y = self.height - self.agent_pos[0] - 1
            self.agent_patch.set_xy((x, y))
        
        # Update title
        self.ax.set_title(f'GridWorld - Step {self.episode_steps}/{self.max_steps}\nPress Q to quit', 
                         fontsize=14, fontweight='bold')
        
        # Redraw only the changed elements
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def render(self, block=True):
        """
        Display the current grid state (legacy method - creates new plot each time).
        
        Args:
            block: If True, program waits for window to close. 
                   If False, continues immediately (use plt.pause() for brief display)
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Draw grid lines
        for i in range(self.height + 1):
            ax.axhline(i, color='black', linewidth=1)
        for i in range(self.width + 1):
            ax.axvline(i, color='black', linewidth=1)
        
        # Draw obstacles (gray)
        for obs in self.obstacles:
            x = obs[1]
            y = self.height - obs[0] - 1
            rect = Rectangle((x, y), 1, 1, facecolor='gray', 
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        
        # Draw goals (red)
        for goal in self.goals:
            x = goal[1]
            y = self.height - goal[0] - 1
            rect = Rectangle((x, y), 1, 1, facecolor='red', 
                           edgecolor='darkred', linewidth=3, label='Goal')
            ax.add_patch(rect)
        
        # Draw agent (green)
        if self.agent_pos is not None:
            x = self.agent_pos[1]
            y = self.height - self.agent_pos[0] - 1
            rect = Rectangle((x, y), 1, 1, facecolor='green', 
                           edgecolor='darkgreen', linewidth=3, label='Agent')
            ax.add_patch(rect)
        
        # Configure plot
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title(f'GridWorld - Step {self.episode_steps}/{self.max_steps}\nPress Q to quit', 
                    fontsize=14, fontweight='bold')
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', 
                 bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if block:
            plt.show()
        else:
            plt.pause(0.1)
        
        return fig
    
    def close(self):
        """Close all matplotlib windows."""
        plt.close('all')
        self.fig = None
        self.ax = None
        self.agent_patch = None
    
    def _is_valid_position(self, row, col):
        """Check if position is within grid boundaries."""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def _min_distance_to_goal(self):
        """Calculate minimum Manhattan distance to any goal."""
        distances = [abs(self.agent_pos[0] - goal[0]) + 
                    abs(self.agent_pos[1] - goal[1]) 
                    for goal in self.goals]
        return min(distances)


def run_episode(env, agent_policy='random', render=False, seed=None, delay=0.5):
    """
    Run a single episode with smooth animation.
    
    Args:
        env: GridWorld environment
        agent_policy: 'random' or custom policy function
        render: Whether to render each step
        seed: Random seed
        delay: Delay between steps when rendering (seconds)
    
    Returns:
        Dictionary with episode results, or None if interrupted
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)  # FIX: Also seed numpy
    
    observation, info = env.reset(seed=seed)
    total_reward = 0
    done = False
    
    if render:
        env.render_init()  # Initialize once
        env.render_update()  # Show initial position
        plt.pause(delay)
    
    try:
        while not done:
            # Choose action
            if agent_policy == 'random':
                action = random.randint(0, 3)
            else:
                action = agent_policy(observation, info)
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if render:
                env.render_update()  # Update only agent position
                plt.pause(delay)
                # Check if window was closed
                if not plt.get_fignums():
                    print("\nWindow closed by user")
                    return None
        
        if render:
            plt.pause(1.0)  # Pause at end to see final state
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl+C)")
        plt.close('all')
        raise  # Re-raise to stop all examples
    
    finally:
        if render:
            plt.ioff()  # Turn off interactive mode
    
    return {
        'total_reward': total_reward,
        'steps': env.episode_steps,
        'success': terminated
    }


# Example usage
if __name__ == "__main__":
    
    print("GridWorld Environment - Smooth Animation - Press Ctrl+C at any time to stop\n")
    
    try:
        # Example 1: Simple environment with 1 goal (backward compatible - square grid)
        print("Example 1: Basic GridWorld (5x5)")
        env = GridWorldEnv(
            grid_size=5,
            goals=[4, 4],
            start_pos=[0, 0]
        )
        result = run_episode(env, render=True, delay=0.2, seed=42)
        if result:
            print(f"Result: {result}\n")
        env.close()
        
        # Example 2: Rectangular grid (10 wide x 5 tall)
        print("Example 2: Rectangular Grid (10x5)")
        env = GridWorldEnv(
            width=10,
            height=5,
            goals=[[4, 9]],
            start_pos=[0, 0]
        )
        result = run_episode(env, render=True, delay=0.2, seed=42)
        if result:
            print(f"Result: {result}\n")
        env.close()
        
        # Example 3: With obstacles (backward compatible)
        print("Example 3: With Obstacles (8x8)")
        env = GridWorldEnv(
            grid_size=8,
            goals=[[7, 7]],
            start_pos=[0, 0],
            obstacles=[[2, 2], [2, 3], [2, 4], [5, 5], [5, 6]]
        )
        result = run_episode(env, render=True, delay=0.2, seed=42)
        if result:
            print(f"Result: {result}\n")
        env.close()
        
        # Example 4: Rectangular with obstacles (15 wide x 8 tall)
        print("Example 4: Wide Rectangular Grid (15x8)")
        env = GridWorldEnv(
            width=15,
            height=8,
            goals=[[7, 14], [0, 14]],
            start_pos=None,  # Random start
            obstacles=[[3, i] for i in range(7)] + [[5, i] for i in range(8, 15)],
            reward_goal=100.0,
            reward_step=-1.0,
            max_steps=100
        )
        result = run_episode(env, render=True, delay=0.1, seed=42)
        if result:
            print(f"Result: {result}")
        env.close()
    
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user")
    
    finally:
        plt.close('all')
        print("\nAll windows closed. Program ended.")