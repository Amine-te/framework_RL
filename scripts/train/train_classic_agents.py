"""
Train Classic RL Agents
=======================

Train and compare classic RL algorithms:
- Policy Iteration (PI) - Model-based
- Value Iteration (VI) - Model-based
- Monte Carlo (MC) - Model-free
- Q-Learning (QL) - Model-free

Usage:
------
python train/train_classic_agents.py --agent all
python train/train_classic_agents.py --agent PI VI
python train/train_classic_agents.py --agent QL --episodes 2000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from src.gridworld import GridWorldEnv
from scripts.agents import (
    PolicyIterationAgent,
    ValueIterationAgent,
    MonteCarloAgent,
    QLearningAgent
)
from scripts.utils.visualization import compare_agents, print_comparison_table


def main():
    """Main function to run agents individually or compare them"""
    
    parser = argparse.ArgumentParser(description='Train and evaluate RL agents on GridWorld')
    parser.add_argument('--agent', nargs='+', default=['all'],
                       choices=['PI', 'VI', 'MC', 'QL', 'all'],
                       help='Which agent(s) to run (default: all)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes for MC and QL (default: 1000)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--grid-size', type=int, default=5,
                       help='Size of the grid (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create environment
    env = GridWorldEnv(
        grid_size=args.grid_size,
        goals=[[args.grid_size-1, args.grid_size-1]],
        start_pos=None,  # Random start
        reward_goal=10.0,
        reward_step=-1,
        max_steps=50
    )
    
    print("\n" + "="*80)
    print("GRIDWORLD REINFORCEMENT LEARNING AGENTS")
    print("="*80)
    print(f"Environment: {args.grid_size}x{args.grid_size} grid")
    print(f"Goals: {env.goals}")
    print(f"Training episodes (MC/QL): {args.episodes}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Random seed: {args.seed}")
    print("="*80)
    
    # Determine which agents to run
    if 'all' in args.agent:
        agents_to_run = ['PI', 'VI', 'MC', 'QL']
    else:
        agents_to_run = args.agent
    
    # Dictionary to store results
    agents_results = {}
    
    # Train and evaluate each agent
    if 'PI' in agents_to_run:
        agent = PolicyIterationAgent(env)
        agent.train()
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        agents_results['Policy Iteration'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")

    if 'VI' in agents_to_run:
        agent = ValueIterationAgent(env)
        agent.train()
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        agents_results['Value Iteration'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")
    
    if 'MC' in agents_to_run:
        agent = MonteCarloAgent(env)
        train_results = agent.train(num_episodes=args.episodes, seed=args.seed)
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        # Include training data for learning curves
        eval_results['returns'] = train_results['episode_returns']
        agents_results['Monte Carlo'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")
    
    if 'QL' in agents_to_run:
        agent = QLearningAgent(env)
        train_results = agent.train(num_episodes=args.episodes, seed=args.seed)
        eval_results = agent.evaluate(num_episodes=args.eval_episodes, seed=args.seed)
        # Include training data for learning curves
        eval_results['returns'] = train_results['episode_returns']
        agents_results['Q-Learning'] = eval_results
        print(f"\n{agent.name} Evaluation:")
        print(f"  Success Rate: {eval_results['success_rate']:.1f}%")
        print(f"  Average Return: {eval_results['avg_return']:.2f}")
        print(f"  Average Length: {eval_results['avg_length']:.1f} steps")
    
    # Print comparison table if multiple agents
    if len(agents_results) > 1:
        print_comparison_table(agents_results)
        
        # Create and save comparison plot
        fig = compare_agents(agents_results, 
                            grid_size=args.grid_size, 
                            episodes=args.episodes,
                            agents_list=agents_to_run)
        import matplotlib.pyplot as plt
        plt.close(fig)  # Close the figure to free memory
    elif len(agents_results) == 1:
        # Save plot even for single agent if it has training data
        name = list(agents_results.keys())[0]
        if 'returns' in agents_results[name]:
            fig = compare_agents(agents_results,
                                grid_size=args.grid_size,
                                episodes=args.episodes,
                                agents_list=agents_to_run)
            import matplotlib.pyplot as plt
            plt.close(fig)


if __name__ == "__main__":
    main()

