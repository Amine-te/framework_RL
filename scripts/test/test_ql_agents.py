"""
Test Q-Learning Agents
======================

Load trained agents and create GIF visualizations of their behavior.

Usage:
    # Test all available agents
    python scripts/test/test_ql_agents.py --all

    # Test specific agents
    python scripts/test/test_ql_agents.py --agents g5_a0.3_gm0.99 g7_a0.3_gm0.99

    # Custom output directory
    python scripts/test/test_ql_agents.py --all --output-dir custom_gifs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import io

from src.gridworld import GridWorldEnv
from scripts.train.train_q_learning import QLearningAgent, create_environment


def parse_agent_filename(filename):
    """
    Parse agent filename to extract parameters.
    
    Expected format: q_learning_ep{episodes}_alpha{alpha}_gamma{gamma}_grid{size}.pkl
    Example: q_learning_ep1000_alpha0.1_gamma0.9_grid7.pkl
    
    Returns:
        dict with keys: episodes, alpha, gamma, grid_size, or None if parsing fails
    """
    pattern = r'q_learning_ep(\d+)_alpha([\d.]+)_gamma([\d.]+)_grid(\d+)\.pkl'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'episodes': int(match.group(1)),
            'alpha': float(match.group(2)),
            'gamma': float(match.group(3)),
            'grid_size': int(match.group(4))
        }
    return None


def find_agent_files(agents_dir, agent_patterns=None):
    """
    Find agent files matching the given patterns or return all if None.
    
    Args:
        agents_dir: Directory containing agent files
        agent_patterns: List of patterns like ['g5_a0.3_gm0.99'] or None for all
        
    Returns:
        List of (filepath, params_dict) tuples
    """
    agents_dir = Path(agents_dir)
    if not agents_dir.exists():
        print(f"Error: Agents directory not found: {agents_dir}")
        return []
    
    all_files = list(agents_dir.glob('q_learning_*.pkl'))
    
    if agent_patterns is None:
        # Return all files
        results = []
        for filepath in all_files:
            params = parse_agent_filename(filepath.name)
            if params:
                results.append((filepath, params))
        return results
    
    # Filter by patterns
    results = []
    for pattern in agent_patterns:
        # Parse pattern like "g5_a0.3_gm0.99" -> grid=5, alpha=0.3, gamma=0.99
        pattern_match = re.match(r'g(\d+)_a([\d.]+)_gm([\d.]+)', pattern)
        if not pattern_match:
            print(f"Warning: Invalid pattern format '{pattern}'. Expected format: g{{size}}_a{{alpha}}_gm{{gamma}}")
            continue
        
        target_grid = int(pattern_match.group(1))
        target_alpha = float(pattern_match.group(2))
        target_gamma = float(pattern_match.group(3))
        
        # Find matching files
        for filepath in all_files:
            params = parse_agent_filename(filepath.name)
            if params and (params['grid_size'] == target_grid and 
                          abs(params['alpha'] - target_alpha) < 1e-6 and
                          abs(params['gamma'] - target_gamma) < 1e-6):
                results.append((filepath, params))
                break
        else:
            print(f"Warning: No agent found matching pattern '{pattern}'")
    
    return results


def render_frame(env, fig, ax):
    """
    Render a single frame of the environment.
    
    Returns:
        PIL Image of the current frame
    """
    ax.clear()
    
    # Draw grid lines
    for i in range(env.height + 1):
        ax.axhline(i, color='black', linewidth=1)
    for i in range(env.width + 1):
        ax.axvline(i, color='black', linewidth=1)
    
    # Draw obstacles (gray)
    for obs in env.obstacles:
        x = obs[1]
        y = env.height - obs[0] - 1
        rect = Rectangle((x, y), 1, 1, facecolor='gray', 
                       edgecolor='black', linewidth=2)
        ax.add_patch(rect)
    
    # Draw goals (red)
    for goal in env.goals:
        x = goal[1]
        y = env.height - goal[0] - 1
        rect = Rectangle((x, y), 1, 1, facecolor='red', 
                       edgecolor='darkred', linewidth=3, label='Goal')
        ax.add_patch(rect)
    
    # Draw agent (green)
    if env.agent_pos is not None:
        x = env.agent_pos[1]
        y = env.height - env.agent_pos[0] - 1
        rect = Rectangle((x, y), 1, 1, facecolor='green', 
                       edgecolor='darkgreen', linewidth=3, label='Agent')
        ax.add_patch(rect)
    
    # Configure plot
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.set_title(f'GridWorld - Step {env.episode_steps}/{env.max_steps}', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    return img


def run_episode_with_gif(env, agent, seed=42):
    """
    Run a single episode and capture frames for GIF creation.
    
    Args:
        env: GridWorld environment
        agent: QLearningAgent instance
        seed: Random seed
        
    Returns:
        tuple: (frames_list, episode_result_dict)
    """
    # Set agent to greedy mode (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    np.random.seed(seed)
    
    observation, info = env.reset(seed=seed)
    frames = []
    total_reward = 0
    done = False
    
    # Initialize figure for rendering
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.ion()
    
    # Capture initial frame
    frames.append(render_frame(env, fig, ax))
    
    try:
        while not done:
            # Choose action (greedy)
            action = agent.choose_action(observation)
            
            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Capture frame
            frames.append(render_frame(env, fig, ax))
            
            # Safety check for max frames
            if len(frames) > env.max_steps + 10:
                break
        
        # Add a few extra frames at the end to show final state
        for _ in range(3):
            frames.append(render_frame(env, fig, ax))
    
    finally:
        plt.close(fig)
        agent.epsilon = original_epsilon
    
    return frames, {
        'total_reward': total_reward,
        'steps': env.episode_steps,
        'success': terminated,
        'truncated': truncated
    }


def create_gif(frames, output_path, duration=200):
    """
    Create a GIF from a list of PIL Images.
    
    Args:
        frames: List of PIL Images
        output_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
    """
    if not frames:
        print(f"Warning: No frames to create GIF for {output_path}")
        return
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as GIF
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"  ✓ GIF saved: {output_path}")


def test_agent(agent_path, params, output_dir, seed=42):
    """
    Test a single agent and create a GIF.
    
    Args:
        agent_path: Path to the agent pickle file
        params: Dictionary with agent parameters
        output_dir: Directory to save the GIF
        seed: Random seed for episode
        
    Returns:
        Episode result dictionary
    """
    print(f"\nTesting agent: {agent_path.name}")
    print(f"  Grid: {params['grid_size']}, Alpha: {params['alpha']}, Gamma: {params['gamma']}")
    
    try:
        # Load agent
        agent = QLearningAgent.load(str(agent_path))
        
        # Create environment
        env = create_environment(params['grid_size'])
        
        # Run episode and capture frames
        frames, result = run_episode_with_gif(env, agent, seed=seed)
        
        # Create output filename
        output_filename = f"test_ql_grid{params['grid_size']}_alpha{params['alpha']}_gamma{params['gamma']}.gif"
        output_path = Path(output_dir) / output_filename
        
        # Create GIF
        create_gif(frames, output_path)
        
        # Print result
        status = "SUCCESS" if result['success'] else ("TRUNCATED" if result['truncated'] else "UNKNOWN")
        print(f"  Episode result: {status}")
        print(f"    Steps: {result['steps']}, Reward: {result['total_reward']:.2f}")
        
        env.close()
        return result
        
    except Exception as e:
        print(f"  ✗ Error testing agent: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Test Q-Learning agents and create GIF visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all available agents
  python scripts/test/test_ql_agents.py --all

  # Test specific agents by pattern
  python scripts/test/test_ql_agents.py --agents g5_a0.3_gm0.99 g7_a0.3_gm0.99

  # Custom output directory
  python scripts/test/test_ql_agents.py --all --output-dir custom_gifs
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all available agents in results/agents/'
    )
    
    parser.add_argument(
        '--agents',
        nargs='+',
        help='Test specific agents by pattern (e.g., g5_a0.3_gm0.99 g7_a0.3_gm0.99)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/gifs',
        help='Output directory for GIFs (default: results/gifs)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for episode (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.agents:
        parser.error("Must specify either --all or --agents")
    
    # Find agent files
    project_root = Path(__file__).parent.parent.parent
    agents_dir = project_root / 'results' / 'agents'
    
    if args.all:
        agent_files = find_agent_files(agents_dir, None)
    else:
        agent_files = find_agent_files(agents_dir, args.agents)
    
    if not agent_files:
        print("No agent files found to test.")
        return
    
    print("=" * 70)
    print("Q-LEARNING AGENT TESTING")
    print("=" * 70)
    print(f"Found {len(agent_files)} agent(s) to test")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    
    # Test each agent
    results = []
    for agent_path, params in agent_files:
        result = test_agent(agent_path, params, args.output_dir, seed=args.seed)
        if result:
            results.append((agent_path.name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if results:
        success_count = sum(1 for _, r in results if r['success'])
        print(f"Tested {len(results)} agent(s)")
        print(f"Successful episodes: {success_count}/{len(results)}")
        print(f"\nGIFs saved to: {Path(args.output_dir).absolute()}")
    else:
        print("No agents were successfully tested.")


if __name__ == '__main__':
    main()

