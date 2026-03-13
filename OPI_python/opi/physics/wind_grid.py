"""
Fourier grid transformation utilities.

Coordinate transformation between geographic (x,y) and wind-aligned (s,t) grids.
Matches MATLAB's windGrid.m functionality.
"""

import numpy as np


def wind_grid(x, y, azimuth):
    """
    Conversion between geographic and wind grids.
    
    Sets up the transformation of a geographic grid, with x,y grid vectors,
    to a wind grid, with s,t grid vectors. Geographic grids use the meshgrid 
    format, and wind grids use the ndgrid format.
    
    Note that x,y,z and s,t,z are both right-handed coordinate frames, as 
    required to ensure correct calculation of the Coriolis effect.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors indicating coordinates for geographic grids,
        with x and y oriented in the east and north directions, respectively
        (vectors with lengths nX and nY, m).
    azimuth : float
        Wind direction (down wind) in degrees from North.
    
    Returns
    -------
    Sxy, Txy : ndarray
        Grids containing the s and t coordinates for nodes in a geographic grid.
        These grids provide the basis for interpolating field variables from 
        the wind-grid solution, back onto grid nodes (matrices, nY x nX, m).
    s, t : ndarray
        Grid vectors indicating coordinates for wind grids, with s oriented 
        in the down-wind direction, and t oriented 90 degrees counterclockwise 
        from the s direction (s,t,z is right handed).
    Xst, Yst : ndarray
        Grids containing the x and y coordinates for nodes in a wind grid.
    """
    # Grid spacing
    dX = x[1] - x[0]
    dY = y[1] - y[0]
    
    # Calculate s,t coordinates for the x,y nodes of geographic grid
    # (s,t,z is right handed)
    X, Y = np.meshgrid(x, y)
    azimuth_rad = np.deg2rad(azimuth)
    Sxy = X * np.sin(azimuth_rad) + Y * np.cos(azimuth_rad)
    Txy = -X * np.cos(azimuth_rad) + Y * np.sin(azimuth_rad)
    
    # Calculate initial estimate for node spacing for the new grid
    # Using ellipse equation to find nodal spacing
    dS = np.sqrt(1.0 / ((np.sin(azimuth_rad) / dX)**2 + (np.cos(azimuth_rad) / dY)**2))
    dT = np.sqrt(1.0 / ((np.sin(azimuth_rad + np.pi/2) / dX)**2 + 
                        (np.cos(azimuth_rad + np.pi/2) / dY)**2))
    
    # Calculate s and t vectors for wind grid
    s_min = np.min(Sxy)
    s_max = np.max(Sxy)
    t_min = np.min(Txy)
    t_max = np.max(Txy)
    
    # Adjust spacing to fit exact number of points
    dS = (s_max - s_min) / np.ceil((s_max - s_min) / dS)
    dT = (t_max - t_min) / np.ceil((t_max - t_min) / dT)
    
    s = np.arange(s_min, s_max + dS/2, dS)
    t = np.arange(t_min, t_max + dT/2, dT)
    
    # Calculate x,y coordinates for the s,t nodes of wind grid
    # (x,y,z and s,t,z are right-handed)
    S, T = np.meshgrid(s, t, indexing='ij')  # ndgrid format
    Xst = S * np.sin(azimuth_rad) - T * np.cos(azimuth_rad)
    Yst = S * np.cos(azimuth_rad) + T * np.sin(azimuth_rad)
    
    return Sxy, Txy, s, t, Xst, Yst
