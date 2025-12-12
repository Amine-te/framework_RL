"""
Common helper functions used across the framework
"""

import numpy as np
import random


def moving_average(data, window=100):
    """
    Calculate moving average for smoothing data.
    
    Args:
        data: List or array of values
        window: Window size for moving average
        
    Returns:
        Smoothed data array
    """
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def get_random_goal(grid_size, avoid_positions=None):
    """
    Generate a random goal position that doesn't conflict with obstacles or start position.
    
    Args:
        grid_size: Size of the grid
        avoid_positions: List of positions to avoid (e.g., obstacles, start position)
        
    Returns:
        [row, col] goal position
    """
    if avoid_positions is None:
        avoid_positions = []
    
    while True:
        goal = [np.random.randint(0, grid_size), np.random.randint(0, grid_size)]
        if goal not in avoid_positions:
            return goal

