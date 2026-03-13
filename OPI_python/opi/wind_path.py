"""
Wind path calculation module.

Calculates wind paths passing through a reference point.
Matches MATLAB's windPath.m functionality.
"""

import numpy as np
from typing import Tuple, Optional


def _line_intersection(p1: np.ndarray, p2: np.ndarray, 
                       p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate intersection point of two line segments.
    
    Parameters
    ----------
    p1, p2 : ndarray
        First line segment endpoints (2D)
    p3, p4 : ndarray
        Second line segment endpoints (2D)
    
    Returns
    -------
    ndarray or None
        Intersection point [x, y] or None if no intersection
    """
    # Line 1: p1 + t*(p2-p1)
    # Line 2: p3 + s*(p4-p3)
    # Solve: p1 + t*(p2-p1) = p3 + s*(p4-p3)
    
    d1 = p2 - p1
    d2 = p4 - p3
    
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return None
    
    dp = p3 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / denom
    s = (dp[0] * d1[1] - dp[1] * d1[0]) / denom
    
    # Check if intersection is within both segments
    if -1e-10 <= t <= 1.00000001 and -1e-10 <= s <= 1.00000001:
        return p1 + t * d1
    
    return None


def _polyxpoly(x1: np.ndarray, y1: np.ndarray, 
               x2: np.ndarray, y2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find intersection points of two polylines.
    
    MATLAB's polyxpoly equivalent.
    
    Parameters
    ----------
    x1, y1 : ndarray
        First polyline coordinates
    x2, y2 : ndarray
        Second polyline coordinates
    
    Returns
    -------
    x_int, y_int : ndarray
        Intersection point coordinates
    """
    intersections_x = []
    intersections_y = []
    
    # Check each segment of polyline 1 against each segment of polyline 2
    for i in range(len(x1) - 1):
        p1 = np.array([x1[i], y1[i]])
        p2 = np.array([x1[i+1], y1[i+1]])
        
        for j in range(len(x2) - 1):
            p3 = np.array([x2[j], y2[j]])
            p4 = np.array([x2[j+1], y2[j+1]])
            
            intersect = _line_intersection(p1, p2, p3, p4)
            if intersect is not None:
                intersections_x.append(intersect[0])
                intersections_y.append(intersect[1])
    
    return np.array(intersections_x), np.array(intersections_y)


def wind_path(x_path0: float, y_path0: float, azimuth: float, 
              x: np.ndarray, y: np.ndarray,
              xy_limits: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate wind path passing through a reference point.
    
    Matches MATLAB's windPath.m functionality.
    
    Parameters
    ----------
    x_path0, y_path0 : float
        Reference point coordinates defining the wind path
    azimuth : float
        Wind direction in degrees (measured clockwise from north)
    x, y : ndarray
        Grid vectors for topography grid
    xy_limits : ndarray, optional
        4-component vector [xMin, xMax, yMin, yMax] defining a smaller
        domain for display purposes. If not provided, uses limits of x and y.
    
    Returns
    -------
    x_path, y_path : ndarray
        Easting and northing for path, bounded by grid limits
    s_path : ndarray
        Distance (meters) along path, with x0,y0 point set to zero,
        and distance increasing in wind direction
    s_limits : ndarray
        Vector with [sMin, sMax] values where xy_limits intersects 
        the wind path
    
    Raises
    ------
    ValueError
        If specified path origin lies outside model domain
    """
    # Set xy_limits
    if xy_limits is not None:
        if len(xy_limits) != 4:
            raise ValueError('Input argument xy_limits must be a 4 component vector.')
    else:
        # Set to limits implied by x and y if not present
        xy_limits = np.array([np.min(x), np.max(x), np.min(y), np.max(y)])
    
    # Calculate step size dS as equal to grid size along wind path
    # dS_path is calculated using ellipse equation:
    # 1/dS^2 = (sin(azimuth)/dX)^2 + (cos(azimuth)/dY)^2
    d_x = x[1] - x[0]
    d_y = y[1] - y[0]
    d_s_path = np.sqrt(1.0 / ((np.sin(np.radians(azimuth)) / d_x)**2 + 
                               (np.cos(np.radians(azimuth)) / d_y)**2))
    
    # Find maximum distance between reference point and grid corners
    x_box = np.array([x[0], x[-1], x[-1], x[0], x[0]])
    y_box = np.array([y[0], y[0], y[-1], y[-1], y[0]])
    d_max = np.max(np.sqrt((x_box - x_path0)**2 + (y_box - y_path0)**2))
    
    # Construct path that extends beyond the bounding box
    x_path_line = x_path0 + np.sin(np.radians(azimuth)) * 1.01 * np.array([-d_max, d_max])
    y_path_line = y_path0 + np.cos(np.radians(azimuth)) * 1.01 * np.array([-d_max, d_max])
    
    # Find intersection with bounding box
    x_path_int, y_path_int = _polyxpoly(x_path_line, y_path_line, x_box, y_box)
    
    # Throw error if only one intersection between bounding box and path
    if len(x_path_int) < 2:
        raise ValueError('Specified path origin must lie inside model domain.')
    
    # Ensure that x_path, y_path are ordered from start to finish
    # Check if direction is correct using dot product
    path_direction = np.array([x_path_int[1] - x_path_int[0], 
                                y_path_int[1] - y_path_int[0]])
    wind_direction = np.array([np.sin(np.radians(azimuth)), 
                                np.cos(np.radians(azimuth))])
    
    if np.dot(path_direction, wind_direction) < 0:
        # Reverse order
        x_path_int = x_path_int[[1, 0]]
        y_path_int = y_path_int[[1, 0]]
    
    # Calculate distance along path, relative to x0, y0, and positive in wind direction
    if abs(x_path_int[1] - x_path_int[0]) > abs(y_path_int[1] - y_path_int[0]):
        s_path_endpoints = (x_path_int - x_path0) / np.sin(np.radians(azimuth))
    else:
        s_path_endpoints = (y_path_int - y_path0) / np.cos(np.radians(azimuth))
    
    # Set nodes with a spacing close to d_s, and with first and last nodes
    # located on the windward and leeward sides of the grid
    n_s = int(np.floor((s_path_endpoints[1] - s_path_endpoints[0]) / d_s_path)) + 1
    
    # Calculate x, y coordinates for nodes along wind path
    x_path = np.linspace(x_path_int[0], x_path_int[1], n_s)
    y_path = np.linspace(y_path_int[0], y_path_int[1], n_s)
    
    # s_path coordinates along wind path, with s_path = 0 at reference point
    s_path = np.linspace(s_path_endpoints[0], s_path_endpoints[1], n_s)
    
    # Calculate s_limits
    # Consider case where xy_limits is specified
    x_limits_box = np.array([xy_limits[0], xy_limits[1], xy_limits[1], xy_limits[0], xy_limits[0]])
    y_limits_box = np.array([xy_limits[2], xy_limits[2], xy_limits[3], xy_limits[3], xy_limits[2]])
    
    x_limits_int, y_limits_int = _polyxpoly(x_path, y_path, x_limits_box, y_limits_box)
    
    if len(x_limits_int) < 2:
        # No intersection with xy_limits box, so set s_limits to ends of s_path
        s_limits = np.array([s_path[0], s_path[-1]])
    else:
        # Calculate s values at intersection points
        if abs(x_limits_int[1] - x_limits_int[0]) > abs(y_limits_int[1] - y_limits_int[0]):
            s_limits = np.interp(x_limits_int, x_path, s_path)
        else:
            s_limits = np.interp(y_limits_int, y_path, s_path)
        
        if s_limits[1] - s_limits[0] < 0:
            s_limits = s_limits[[1, 0]]
    
    return x_path, y_path, s_path, s_limits
