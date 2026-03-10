"""
Orographic Lifting Calculations and Wind Path Utilities

This module provides functions for calculating orographic lifting effects
and generating wind path trajectories for the OPI model.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .fourier_solution import wind_grid


def lifting(x, y, h_grid, p_grid, azimuth, U, tau_f, ij_catch, ptr_catch):
    """
    Calculate maximum upwind lifting for sample locations.
    
    This function computes the orographic lifting experienced by air parcels
    as they approach sample locations, accounting for precipitation-weighted
    averaging for catchment-integrated samples.
    
    Parameters
    ----------
    x, y : ndarray
        Grid coordinate vectors (m)
    h_grid : ndarray
        Topography grid in geographic coordinates (m). Shape (ny, nx) where
        ny = len(y) and nx = len(x).
    p_grid : ndarray
        Precipitation rate grid (kg/m²/s). Same shape as h_grid.
    azimuth : float
        Wind direction in degrees from North (downwind direction)
    U : float
        Wind speed (m/s)
    tau_f : float
        Precipitation fall time (s). Characteristic time for hydrometeors
        to fall from condensation level to surface.
    ij_catch : ndarray
        Catchment indices. Array of linear indices into the grid for each
        catchment node.
    ptr_catch : ndarray
        Catchment pointers. Array where ptr_catch[i] gives the starting
        index in ij_catch for catchment i, and ptr_catch[i+1] gives the
        ending index (exclusive).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'lift_max' : ndarray
            Maximum upwind lifting elevation for each catchment (m)
        - 'elevation' : ndarray
            Surface elevation at each catchment (m)
        - 'subsidence' : ndarray
            Subsidence flag indicating if the location is in a
            subsidence zone (True) or lifting zone (False)
            
    Notes
    -----
    The lifting calculation accounts for:
    1. Upwind fetch: Air parcels are lifted as they encounter topography
    2. Fall time delay: Hydrometeors formed at the lifting maximum take
       tau_f seconds to reach the surface
    3. Precipitation weighting: For catchment samples, the lifting is
       weighted by the precipitation distribution within the catchment
       
    The maximum upwind elevation is found by tracking back along the wind
    path by a distance U * tau_f and finding the maximum elevation
    encountered along the way.
    
    References
    ----------
    Smith (2006) - Precipitation blocking
    Smith and Barstad (2004) - Linear theory of orographic precipitation
    """
    n_catchments = len(ptr_catch) - 1
    
    # Initialize output arrays
    lift_max = np.zeros(n_catchments)
    elevation = np.zeros(n_catchments)
    subsidence = np.zeros(n_catchments, dtype=bool)
    
    # Transform to wind-aligned coordinates
    grid_info = wind_grid(x, y, azimuth)
    Sxy = grid_info['Sxy']  # Wind-aligned s coordinates on geographic grid
    Txy = grid_info['Txy']  # Wind-aligned t coordinates on geographic grid
    
    # Convert azimuth to radians for calculations
    azimuth_rad = np.deg2rad(azimuth)
    
    # Calculate upwind displacement distance
    # Air parcels travel upwind by U * tau_f during fall time
    delta_s = U * tau_f
    
    # Create interpolators for topography and precipitation
    X, Y = np.meshgrid(x, y)
    
    # Interpolation for topography on geographic grid
    h_interp = RegularGridInterpolator(
        (y, x), h_grid,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    
    # Interpolation for precipitation on geographic grid
    p_interp = RegularGridInterpolator(
        (y, x), p_grid,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    
    # Process each catchment
    for i_catch in range(n_catchments):
        i_start = ptr_catch[i_catch]
        i_end = ptr_catch[i_catch + 1]
        
        # Get indices for this catchment
        catchment_indices = ij_catch[i_start:i_end]
        
        # Convert linear indices to 2D indices
        ny, nx = h_grid.shape
        iy = catchment_indices // nx
        ix = catchment_indices % nx
        
        # Get coordinates and values for catchment nodes
        x_catch = x[ix]
        y_catch = y[iy]
        h_catch = h_grid[iy, ix]
        p_catch = p_grid[iy, ix]
        
        # Get wind-aligned coordinates for catchment nodes
        s_catch = Sxy[iy, ix]
        t_catch = Txy[iy, ix]
        
        # Calculate total precipitation for weighting
        p_total = np.sum(p_catch)
        
        if p_total > 0:
            # Precipitation weights
            weights = p_catch / p_total
        else:
            # Uniform weighting if no precipitation
            weights = np.ones_like(p_catch) / len(p_catch)
        
        # Calculate surface elevation (precipitation-weighted average)
        elevation[i_catch] = np.sum(weights * h_catch)
        
        # Find maximum upwind elevation for each node in catchment
        lift_max_nodes = np.zeros(len(catchment_indices))
        
        for i_node in range(len(catchment_indices)):
            # Starting point (upwind of the node by delta_s)
            s_upwind = s_catch[i_node] - delta_s
            t_node = t_catch[i_node]
            
            # Create path points from upwind to current position
            n_path_points = max(20, int(delta_s / 100))  # At least 20 points
            s_path = np.linspace(s_upwind, s_catch[i_node], n_path_points)
            t_path = np.full_like(s_path, t_node)
            
            # Convert back to geographic coordinates
            # x = s*sin(azimuth) - t*cos(azimuth)
            # y = s*cos(azimuth) + t*sin(azimuth)
            x_path = s_path * np.sin(azimuth_rad) - t_path * np.cos(azimuth_rad)
            y_path = s_path * np.cos(azimuth_rad) + t_path * np.sin(azimuth_rad)
            
            # Interpolate elevations along the path
            points = np.column_stack([y_path, x_path])
            h_path = h_interp(points)
            
            # Maximum elevation along the path
            lift_max_nodes[i_node] = np.max(h_path)
        
        # Precipitation-weighted average of maximum lifting
        lift_max[i_catch] = np.sum(weights * lift_max_nodes)
        
        # Determine if catchment is in subsidence zone
        # Subsidence occurs when the maximum upwind elevation is less than
        # the surface elevation (i.e., the air is descending)
        subsidence[i_catch] = lift_max[i_catch] < elevation[i_catch]
    
    return {
        'lift_max': lift_max,
        'elevation': elevation,
        'subsidence': subsidence
    }


def wind_path(x0, y0, azimuth, x_grid, y_grid, n_points=100):
    """
    Generate points along a wind path starting from (x0, y0).
    
    Creates a trajectory line following the wind direction from a starting
    point, extending across the full extent of the given grid.
    
    Parameters
    ----------
    x0, y0 : float
        Starting coordinates in meters
    azimuth : float
        Wind direction in degrees from North (clockwise). This is the
        direction the wind is blowing toward (downwind direction).
    x_grid, y_grid : ndarray
        Grid vectors for determining the extent of the wind path.
        The path will span from one edge of the grid to the opposite edge.
    n_points : int, optional
        Number of points along the path. Default is 100.
        
    Returns
    -------
    x_path, y_path : ndarray
        Arrays of x and y coordinates along the wind path. The path
        extends from the upwind edge of the grid domain, through the
        starting point (x0, y0), to the downwind edge.
        
    Notes
    -----
    The azimuth follows meteorological convention:
    - 0° = North (wind blowing from South to North)
    - 90° = East (wind blowing from West to East)
    - 180° = South (wind blowing from North to South)
    - 270° = West (wind blowing from East to West)
    
    The wind path extends in both upwind and downwind directions to cover
    the full extent of the grid domain.
    """
    # Convert azimuth to radians
    # Azimuth is measured clockwise from North
    azimuth_rad = np.deg2rad(azimuth)
    
    # Calculate the direction vector (pointing downwind)
    # For azimuth 0° (North): dx = 0, dy = 1
    # For azimuth 90° (East): dx = 1, dy = 0
    dx = np.sin(azimuth_rad)  # East component
    dy = np.cos(azimuth_rad)  # North component
    
    # Determine grid extents
    x_min, x_max = np.min(x_grid), np.max(x_grid)
    y_min, y_max = np.min(y_grid), np.max(y_grid)
    
    # Calculate grid diagonal (maximum possible distance in grid)
    grid_diagonal = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    
    # Create path extending well beyond grid boundaries
    # Start from far upwind and go to far downwind
    s_values = np.linspace(-grid_diagonal, grid_diagonal, n_points)
    
    # Calculate path coordinates
    # s = 0 at (x0, y0), positive s is downwind
    x_path = x0 + s_values * dx
    y_path = y0 + s_values * dy
    
    return x_path, y_path


def compute_cumulative_max_elevation(h_wind):
    """
    Compute cumulative maximum elevation along the wind direction.
    
    This function calculates the maximum elevation encountered upwind
    at each point along the s-direction (wind-aligned) axis. This is
    used to determine the maximum lifting experienced by air parcels
    as they approach each point.
    
    Parameters
    ----------
    h_wind : ndarray
        Topography grid in wind-aligned coordinates (m).
        Shape (n_s, n_t) where:
        - n_s is the number of points in the downwind (s) direction
        - n_t is the number of points in the crosswind (t) direction
        The s dimension should increase in the downwind direction.
        
    Returns
    -------
    h_max : ndarray
        Cumulative maximum elevation array with same shape as h_wind.
        At each point (i, j), h_max[i, j] is the maximum elevation
        encountered at or upwind of that point (indices 0 through i).
        
    Notes
    -----
    The cumulative maximum is computed along axis 0 (the s-direction),
    which represents the wind direction. For each crosswind position t,
    the function finds the maximum elevation encountered at all s positions
    from the upwind edge to the current position.
    
    This is useful for:
    - Identifying crests and ridges that block airflow
    - Calculating the maximum lifting experienced by air parcels
    - Determining zones of subsidence vs. lifting
    """
    # Compute cumulative maximum along the s-direction (axis 0)
    # This gives the maximum elevation encountered upwind at each point
    h_max = np.maximum.accumulate(h_wind, axis=0)
    
    return h_max
