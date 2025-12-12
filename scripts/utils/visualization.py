"""
Visualization utilities for plotting and comparing agents
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def compare_agents(agents_results, save_dir=None, grid_size=None, episodes=None, agents_list=None):
    """
    Create comparison plot for multiple agents and save to file.
    
    Args:
        agents_results: Dictionary mapping agent names to their results
        save_dir: Directory to save the plot
        grid_size: Grid size for filename
        episodes: Number of episodes for filename
        agents_list: List of agent names for filename
        
    Returns:
        matplotlib figure
    """
    if save_dir is None:
        save_dir = Path(__file__).parent.parent.parent / 'results' / 'plots'
    save_dir = Path(save_dir)
    
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
    """
    Print a formatted comparison table of agent results.
    
    Args:
        agents_results: Dictionary mapping agent names to their results
    """
    print("\n" + "="*80)
    print("AGENT COMPARISON TABLE")
    print("="*80)
    print(f"{'Agent':<20} {'Success Rate':<15} {'Avg Return':<15} {'Avg Length':<15}")
    print("-"*80)
    
    for name, results in agents_results.items():
        success_rate = results.get('success_rate', 0.0)
        avg_return = results.get('avg_return', 0.0)
        avg_length = results.get('avg_length', 0.0)
        print(f"{name:<20} {success_rate:<15.1f} {avg_return:<15.2f} {avg_length:<15.1f}")
    
    print("="*80)

