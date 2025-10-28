"""
Modular GridWorld Environment with Smooth Animation
A clean, parametrized framework for reinforcement learning.
Optional gymnasium compatibility if installed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

# Try to import gymnasium, but don't fail if it's not installed
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    gym = None
    spaces = None


# Base class that may or may not inherit from gym.Env
if GYM_AVAILABLE:
    BaseClass = gym.Env
else:
    BaseClass = object


class GridWorldEnv(BaseClass):
    """
    Customizable grid world environment.
    
    **GYMNASIUM-COMPATIBLE**: If gymnasium is installed, this class automatically
    becomes a valid Gym environment with action_space and observation_space.
    
    **BACKWARD-COMPATIBLE**: Existing code that doesn't use gymnasium will work exactly as before.
    
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
        state_type (str): 'absolute' returns [row, col], 'relative' returns [dx, dy]
    """
    
    # Gymnasium metadata
    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, grid_size=5, width=None, height=None, goals=None, start_pos=None, 
                 obstacles=None, reward_goal=10.0, reward_step=-0.1, max_steps=50,
                 state_type='absolute'):
        
        # Initialize parent class if using Gymnasium
        if GYM_AVAILABLE:
            super().__init__()
        
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
        self.state_type = state_type
        
        self.agent_pos = None
        self.episode_steps = 0
        
        # Rendering attributes
        self.fig = None
        self.ax = None
        self.agent_patch = None
        
        # Define Gymnasium spaces if available
        if GYM_AVAILABLE:
            self.action_space = spaces.Discrete(4)
            
            if state_type == 'relative':
                # Relative state: (dx, dy)
                self.observation_space = spaces.Box(
                    low=-max(self.height, self.width),
                    high=max(self.height, self.width),
                    shape=(2,),
                    dtype=np.float32
                )
            else:
                # Absolute state: [row, col]
                self.observation_space = spaces.Box(
                    low=0,
                    high=max(self.height, self.width),
                    shape=(2,),
                    dtype=np.float32
                )
    
    def reset(self, seed=None, options=None):
        """
        Start a new episode.
        
        Returns:
            observation: Agent position (format depends on state_type)
            info: Dictionary with additional information
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Gymnasium-style reset with options
        if GYM_AVAILABLE and hasattr(super(), 'reset'):
            super().reset(seed=seed)
        
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
        
        # Get observation based on state_type
        observation = self._get_observation()
        
        info = {
            'distance_to_goal': self._min_distance_to_goal(),
            'episode_steps': self.episode_steps
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one action.
        
        Returns:
            observation: Agent position (format depends on state_type)
            reward: Reward for this step
            terminated: Whether episode ended (reached goal)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
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
        
        # Get observation based on state_type
        observation = self._get_observation()
        
        info = {
            'distance_to_goal': self._min_distance_to_goal(),
            'episode_steps': self.episode_steps
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get observation based on state_type setting"""
        if self.state_type == 'relative':
            # Relative state: (dx, dy) to nearest goal
            goal_pos = self.goals[0]  # Use first goal
            dx = goal_pos[1] - self.agent_pos[1]
            dy = goal_pos[0] - self.agent_pos[0]
            obs = [dx, dy]
        else:
            # Absolute state: [row, col]
            obs = list(self.agent_pos)
        
        # Convert to numpy array if Gymnasium is available
        if GYM_AVAILABLE:
            return np.array(obs, dtype=np.float32)
        return obs
    
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
        np.random.seed(seed)
    
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
    
    if GYM_AVAILABLE:
        print("✓ Gymnasium detected - This environment is Gym-compatible!")
        print(f"  Action space: Discrete(4)")
        print(f"  Observation space: Box (depends on state_type)\n")
    else:
        print("ℹ Gymnasium not installed - Using standalone mode")
        print("  Install with: pip install gymnasium\n")
    
    try:
        # Example 1: Basic GridWorld (backward compatible)
        print("Example 1: Basic GridWorld (5x5) - Absolute state")
        env = GridWorldEnv(
            grid_size=5,
            goals=[4, 4],
            start_pos=[0, 0]
        )
        result = run_episode(env, render=True, delay=0.2, seed=42)
        if result:
            print(f"Result: {result}\n")
        env.close()
        
        # Example 2: With relative state (for DQN with moving goals)
        print("Example 2: GridWorld with Relative State (dx, dy)")
        env = GridWorldEnv(
            grid_size=7,
            goals=[[6, 6]],
            start_pos=[0, 0],
            obstacles=[[2, 2], [2, 3], [2, 4]],
            state_type='relative'  # NEW: Returns (dx, dy) instead of (row, col)
        )
        
        if GYM_AVAILABLE:
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")
        
        result = run_episode(env, render=True, delay=0.2, seed=42)
        if result:
            print(f"Result: {result}\n")
        env.close()
        
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user")
    
    finally:
        plt.close('all')
        print("\nAll windows closed. Program ended.")