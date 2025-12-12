"""
Value Iteration Agent
=====================

A model-based algorithm that directly computes the optimal value function
by iteratively applying the Bellman optimality equation.

Pros: Often faster than Policy Iteration, simpler implementation
Cons: Requires complete knowledge of environment dynamics
"""

import numpy as np


class ValueIterationAgent:
    """Value Iteration Agent for GridWorld"""
    
    def __init__(self, env, gamma=0.99, theta=1e-6):
        """
        Args:
            env: GridWorld environment
            gamma: Discount factor
            theta: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.grid_size = env.grid_size
        
        # Initialize value function
        self.V = np.zeros((self.grid_size, self.grid_size))
        self.policy = None
        
        self.name = "Value Iteration"
    
    def _get_next_state(self, state, action):
        """Simulate taking an action from a state"""
        row, col = state
        new_row, new_col = row, col
        
        if action == 0:  # UP
            new_row -= 1
        elif action == 1:  # RIGHT
            new_col += 1
        elif action == 2:  # DOWN
            new_row += 1
        elif action == 3:  # LEFT
            new_col -= 1
        
        # Check boundaries
        new_row = max(0, min(self.grid_size - 1, new_row))
        new_col = max(0, min(self.grid_size - 1, new_col))
        
        # Check obstacles
        if [new_row, new_col] in self.env.obstacles:
            return state  # Stay in place if obstacle
        
        return (new_row, new_col)
    
    def _get_reward(self, state, action, next_state):
        """Get reward for transition"""
        if list(next_state) in self.env.goals:
            return self.env.reward_goal
        return self.env.reward_step
    
    def _is_goal_state(self, state):
        """Check if state is a goal state"""
        return list(state) in self.env.goals
    
    def train(self, max_iterations=1000):
        """
        Train the agent using Value Iteration
        
        Args:
            max_iterations: Maximum number of iterations
            
        Returns:
            Training statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Training {self.name}")
        print(f"{'='*60}")
        
        for iteration in range(max_iterations):
            delta = 0
            new_V = np.copy(self.V)
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    state = (row, col)
                    
                    # Skip if goal state
                    if self._is_goal_state(state):
                        continue
                    
                    # Calculate value for each action
                    action_values = []
                    for action in range(4):
                        next_state = self._get_next_state(state, action)
                        reward = self._get_reward(state, action, next_state)
                        value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                        action_values.append(value)
                    
                    # Take maximum value
                    new_V[row, col] = max(action_values)
                    delta = max(delta, abs(new_V[row, col] - self.V[row, col]))
            
            self.V = new_V
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Delta = {delta:.6f}")
            
            if delta < self.theta:
                print(f"Converged after {iteration + 1} iterations!")
                break
        
        # Extract policy from value function
        self._extract_policy()
        
        return {'iterations': iteration + 1, 'delta': delta}
    
    def _extract_policy(self):
        """Extract optimal policy from value function"""
        self.policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                state = (row, col)
                
                if self._is_goal_state(state):
                    continue
                
                # Find best action
                action_values = []
                for action in range(4):
                    next_state = self._get_next_state(state, action)
                    reward = self._get_reward(state, action, next_state)
                    value = reward + self.gamma * self.V[next_state[0], next_state[1]]
                    action_values.append(value)
                
                self.policy[row, col] = np.argmax(action_values)
    
    def get_action(self, observation):
        """Get action for a given observation"""
        row, col = observation
        return self.policy[int(row), int(col)]
    
    def evaluate(self, num_episodes=100, seed=42):
        """Evaluate the learned policy"""
        returns = []
        lengths = []
        successes = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset(seed=seed + episode)
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            returns.append(total_reward)
            lengths.append(self.env.episode_steps)
            successes.append(terminated)
        
        return {
            'returns': returns,
            'lengths': lengths,
            'successes': successes,
            'success_rate': np.mean(successes) * 100,
            'avg_return': np.mean(returns),
            'avg_length': np.mean(lengths)
        }

