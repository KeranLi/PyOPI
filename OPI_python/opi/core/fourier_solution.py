"""
Fourier Solution for Linearized Euler Equations

This module calculates the Fourier solution for atmospheric flow over 
topography using the linearized anelastic equations of Durran and Klemp (1982).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def wind_grid(x, y, azimuth):
    """
    Conversion between geographic and wind-aligned coordinate systems.
    
    Transforms a geographic grid (x, y) to a wind-aligned grid (s, t) where
    s is oriented in the downwind direction and t is perpendicular (90° CCW).
    Both (x,y,z) and (s,t,z) are right-handed coordinate systems.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors for geographic coordinates (m)
    azimuth : float
        Wind direction in degrees from North (downwind)
        
    Returns
    -------
    dict : Dictionary containing:
        - 'Sxy', 'Txy' : Grids of s,t coordinates for geographic grid nodes
        - 's', 't' : Grid vectors for wind-aligned coordinates
        - 'Xst', 'Yst' : Grids of x,y coordinates for wind grid nodes
        - 'd_s', 'd_t' : Grid spacing in s and t directions
        
    Notes
    -----
    Geographic grids use meshgrid format (x varies across columns, y down rows).
    Wind grids use ndgrid format (s varies down rows, t across columns).
    """
    # Grid spacing
    dX = x[1] - x[0]
    dY = y[1] - y[0]
    
    # Create meshgrid for geographic coordinates
    X, Y = np.meshgrid(x, y)
    
    # Convert azimuth to radians
    azimuth_rad = np.deg2rad(azimuth)
    
    # Calculate s,t coordinates for geographic grid nodes
    # s = x*sin(azimuth) + y*cos(azimuth)  [downwind]
    # t = -x*cos(azimuth) + y*sin(azimuth)  [crosswind, 90° CCW]
    Sxy = X * np.sin(azimuth_rad) + Y * np.cos(azimuth_rad)
    Txy = -X * np.cos(azimuth_rad) + Y * np.sin(azimuth_rad)
    
    # Calculate grid spacing for wind grid using ellipse equation
    # This ensures the new grid has similar nodal density
    d_s = np.sqrt(1.0 / ((np.sin(azimuth_rad) / dX)**2 + 
                         (np.cos(azimuth_rad) / dY)**2))
    d_t = np.sqrt(1.0 / ((np.sin(azimuth_rad + np.pi/2) / dX)**2 + 
                         (np.cos(azimuth_rad + np.pi/2) / dY)**2))
    
    # Determine bounds
    s_min = np.min(Sxy)
    s_max = np.max(Sxy)
    t_min = np.min(Txy)
    t_max = np.max(Txy)
    
    # Adjust spacing to fit exact number of points
    d_s = (s_max - s_min) / np.ceil((s_max - s_min) / d_s)
    d_t = (t_max - t_min) / np.ceil((t_max - t_min) / d_t)
    
    # Create wind grid vectors
    s = np.arange(s_min, s_max + d_s/2, d_s)
    t = np.arange(t_min, t_max + d_t/2, d_t)
    
    # Calculate x,y coordinates for wind grid nodes (ndgrid format)
    S, T = np.meshgrid(s, t, indexing='ij')
    Xst = S * np.sin(azimuth_rad) - T * np.cos(azimuth_rad)
    Yst = S * np.cos(azimuth_rad) + T * np.sin(azimuth_rad)
    
    return {
        'Sxy': Sxy,
        'Txy': Txy,
        's': s,
        't': t,
        'Xst': Xst,
        'Yst': Yst,
        'd_s': d_s,
        'd_t': d_t
    }


def fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho):
    """
    Calculate Fourier solution for linearized Euler equations over topography.
    
    Solves the anelastic equations for saturated flow over arbitrary 
    topography using Fourier transform methods.
    
    Parameters
    ----------
    x, y : ndarray
        Geographic grid vectors (m)
    h_grid : ndarray
        Topography grid in geographic coordinates (m)
    U : float
        Base-state wind speed (m/s)
    azimuth : float
        Wind direction in degrees from North
    NM : float
        Saturated buoyancy frequency (rad/s)
    f_c : float
        Coriolis parameter (rad/s)
    h_rho : float
        Density scale height (m)
        
    Returns
    -------
    dict : Dictionary containing:
        - 's', 't' : Wind grid coordinate vectors
        - 'Sxy', 'Txy' : Geographic coordinates in wind system
        - 'h_wind' : Topography in wind coordinates
        - 'k_s', 'k_t' : Wavenumber vectors (rad/m)
        - 'h_hat' : Fourier coefficients of topography
        - 'k_z' : Vertical wavenumber grid
        - 'n_s_pad', 'n_t_pad' : Padded grid dimensions
        
    Notes
    -----
    The solution handles singularities where |U*k_s| ≈ |f_c| using the
    method of Queney (1947).
    
    References
    ----------
    Durran and Klemp (1982) - On the effects of moisture on the 
        Brunt-Väisälä frequency
    Queney (1947) - Theory of perturbations in stratified currents
    """
    # Transform to wind coordinates
    grid_info = wind_grid(x, y, azimuth)
    Sxy = grid_info['Sxy']
    Txy = grid_info['Txy']
    s = grid_info['s']
    t = grid_info['t']
    Xst = grid_info['Xst']
    Yst = grid_info['Yst']
    
    # Interpolate topography to wind grid
    interp_func = RegularGridInterpolator(
        (y, x), h_grid,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    points = np.column_stack([Yst.ravel(), Xst.ravel()])
    h_wind = interp_func(points).reshape(Xst.shape)
    
    # Grid parameters
    d_s = s[1] - s[0]
    n_s, n_t = h_wind.shape
    
    # Add zero padding (recommended: n_s_pad = 2*n_s, n_t_pad = n_t)
    n_s_pad = 2 * n_s
    n_t_pad = n_t
    
    # Calculate wavenumber grids
    i_k_s_most_neg = int(np.ceil(n_s_pad / 2)) + 1
    k_s = np.arange(n_s_pad) / n_s_pad
    k_s[i_k_s_most_neg:n_s_pad] = k_s[i_k_s_most_neg:n_s_pad] - 1
    k_s = 2 * np.pi * k_s / d_s
    d_k_s = k_s[1] - k_s[0]
    
    i_k_t_most_neg = int(np.ceil(n_t_pad / 2)) + 1
    k_t = np.arange(n_t_pad) / n_t_pad
    k_t[i_k_t_most_neg:n_t_pad] = k_t[i_k_t_most_neg:n_t_pad] - 1
    k_t = 2 * np.pi * k_t / (t[1] - t[0])
    
    # Fourier transform of topography
    h_hat = np.fft.fft2(h_wind, s=(n_s_pad, n_t_pad))
    
    # Handle singularities where |U*k_s| ≈ |f_c|
    k_s_col = k_s.reshape(-1, 1)
    denominator = (U * k_s_col)**2 - f_c**2
    
    # Remove excitation at singularities (Queney, 1947)
    i_zero = np.abs(denominator) < (U * d_k_s / 2)**2
    i_zero_broadcast = np.broadcast_to(i_zero, h_hat.shape)
    h_hat[i_zero_broadcast] = 0
    
    # Avoid division by zero
    denominator[denominator == 0] = np.finfo(float).eps
    
    # Calculate vertical wavenumber
    k_z_sq = ((k_s_col**2 + k_t**2) * 
              ((NM**2 - (U * k_s_col)**2) / denominator) - 
              1.0 / (4 * h_rho**2))
    
    k_z = np.sqrt(k_z_sq.astype(np.complex128))
    
    # Set sign of real k_z to match k_s (ensures upward propagation)
    i_neg = (np.real(k_z_sq) > 0) & (k_s_col < 0)
    i_neg_broadcast = np.broadcast_to(i_neg, k_z.shape)
    k_z[i_neg_broadcast] = -k_z[i_neg_broadcast]
    
    return {
        's': s,
        't': t,
        'Sxy': Sxy,
        'Txy': Txy,
        'h_wind': h_wind,
        'k_s': k_s,
        'k_t': k_t,
        'h_hat': h_hat,
        'k_z': k_z,
        'n_s_pad': n_s_pad,
        'n_t_pad': n_t_pad,
        'd_s': d_s
    }
