"""
Functions to calculate wind path trajectories
"""

import numpy as np
from scipy.spatial.distance import cdist


def wind_path(x0, y0, azimuth, x_grid, y_grid):
    """
    Calculate points along a wind path through a grid.
    
    Parameters:
    -----------
    x0, y0 : float
        Starting point coordinates
    azimuth : float
        Wind direction in degrees (measured clockwise from north)
    x_grid, y_grid : array-like
        Grid coordinates
    
    Returns:
    --------
    x_path, y_path : ndarray
        Coordinates along the wind path
    """
    # Convert azimuth from degrees to radians (clockwise from north)
    # Need to convert to mathematical convention (counterclockwise from east)
    angle_rad = np.radians(90 - azimuth)
    
    # Create a direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    # Generate points along the wind direction
    # We'll use the center of the grid as a reference point
    x_center = np.mean(x_grid)
    y_center = np.mean(y_grid)
    
    # Calculate distances from the starting point to the center of the grid
    dist_to_center = np.sqrt((x_center - x0)**2 + (y_center - y0)**2)
    
    # Determine the range of points to cover the entire grid
    max_dim = max(len(x_grid), len(y_grid))
    step_size = min(np.diff(x_grid[:2])[0], np.diff(y_grid[:2])[0]) if len(x_grid) > 1 else np.abs(x_grid[1] - x_grid[0])
    
    # Calculate how many steps we need to go beyond the grid
    n_steps = int(2 * max_dim)
    
    # Generate points along the wind direction
    t_values = np.linspace(-n_steps * step_size, n_steps * step_size, 2*n_steps + 1)
    x_points = x0 + t_values * dx
    y_points = y0 + t_values * dy
    
    # Filter points to only include those near the grid
    x_min, x_max = np.min(x_grid), np.max(x_grid)
    y_min, y_max = np.min(y_grid), np.max(y_grid)
    
    # Calculate buffer zone (approximately 2 grid spacings)
    x_buffer = (x_max - x_min) * 0.1
    y_buffer = (y_max - y_min) * 0.1
    
    # Keep only points within the grid plus buffer
    mask = ((x_points >= x_min - x_buffer) & (x_points <= x_max + x_buffer) &
            (y_points >= y_min - y_buffer) & (y_points <= y_max + y_buffer))
    
    x_path = x_points[mask]
    y_path = y_points[mask]
    
    return x_path, y_path