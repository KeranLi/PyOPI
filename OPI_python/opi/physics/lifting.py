"""
Lifting and Vertical Motion Calculations

Calculates vertical air motion (w) and related quantities
from the Fourier solution.

Reference: Durran and Klemp, 1982
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def calculate_vertical_velocity(k_s, k_t, k_z, h_hat, U, z_levels, z_bar, 
                                gamma_sat, gamma_env, rho0, h_rho):
    """
    Calculate vertical velocity (w) from the Fourier solution.
    
    Parameters
    ----------
    k_s, k_t, k_z : ndarray
        Wavenumber grids (rad/m)
    h_hat : ndarray
        Fourier coefficients of topography
    U : float
        Wind speed (m/s)
    z_levels : ndarray
        Vertical levels to calculate w (m)
    z_bar : ndarray
        Base state elevation vector (m)
    gamma_sat, gamma_env : ndarray
        Lapse rates (K/m)
    rho0 : float
        Surface air density (kg/m^3)
    h_rho : float
        Density scale height (m)
    
    Returns
    -------
    w_grid : ndarray
        Vertical velocity at specified levels (m/s)
    """
    n_s, n_t = len(k_s), len(k_t)
    n_z = len(z_levels)
    
    # Create 3D grid for output
    w_grid = np.zeros((n_z, n_s, n_t))
    
    # Wavenumber matrices
    K_S, K_T = np.meshgrid(k_s, k_t, indexing='ij')
    K_Z = k_z  # Already 2D from fourier_solution
    
    for i_z, z in enumerate(z_levels):
        # Vertical structure function
        # w decays exponentially with height in the troposphere
        vertical_factor = np.exp(-z / (2 * h_rho))
        
        # Calculate w in Fourier space
        # w_hat = i * U * k_s * h_hat * exp(i * k_z * z) * vertical_factor
        w_hat = (1j * U * K_S * h_hat * 
                 np.exp(1j * K_Z * z) * vertical_factor)
        
        # Transform to physical space
        w = np.fft.ifft2(w_hat, s=(n_s, n_t))
        w_grid[i_z, :, :] = np.real(w)
    
    return w_grid


def calculate_lifting_max(w_grid, z_levels, h_grid, s, t, Sxy, Txy):
    """
    Calculate maximum lifting elevation for each surface point.
    
    This identifies the level of maximum vertical velocity above each point.
    
    Parameters
    ----------
    w_grid : ndarray
        Vertical velocity field (z, s, t)
    z_levels : ndarray
        Vertical levels (m)
    h_grid : ndarray
        Surface topography in wind coordinates (m)
    s, t : ndarray
        Wind coordinate vectors
    Sxy, Txy : ndarray
        Transformation grids from geographic to wind coordinates
    
    Returns
    -------
    lift_max : ndarray
        Elevation of maximum lifting (m above sea level)
    """
    n_s, n_t = len(s), len(t)
    lift_max = np.zeros((n_s, n_t))
    
    for i in range(n_s):
        for j in range(n_t):
            # Surface elevation at this point
            h_surf = h_grid[i, j]
            
            # Find levels above surface
            above_surface = z_levels > h_surf
            if not np.any(above_surface):
                lift_max[i, j] = h_surf
                continue
            
            # Find level with maximum w
            w_profile = w_grid[above_surface, i, j]
            z_above = z_levels[above_surface]
            
            if len(w_profile) > 0:
                idx_max = np.argmax(w_profile)
                lift_max[i, j] = z_above[idx_max]
            else:
                lift_max[i, j] = h_surf
    
    return lift_max


def calculate_streamlines(x, y, u_grid, v_grid, w_grid=None, 
                         start_points=None, n_points=100, max_steps=1000):
    """
    Calculate 3D streamlines from velocity field.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors (m)
    u_grid, v_grid : ndarray
        Horizontal velocity components (m/s)
    w_grid : ndarray, optional
        Vertical velocity component (m/s)
    start_points : ndarray, optional
        Starting points for streamlines (n_points x 3 array)
    n_points : int
        Number of streamlines to calculate if start_points not given
    max_steps : int
        Maximum integration steps
    
    Returns
    -------
    streamlines : list
        List of streamline coordinates (each is n x 3 array)
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolators
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    interp_u = RegularGridInterpolator((x, y), u_grid, 
                                        bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((x, y), v_grid,
                                        bounds_error=False, fill_value=None)
    
    # Generate starting points if not provided
    if start_points is None:
        # Start at multiple locations at z=1000m
        x_starts = np.linspace(np.min(x), np.max(x), int(np.sqrt(n_points)))
        y_starts = np.linspace(np.min(y), np.max(y), int(np.sqrt(n_points)))
        X_start, Y_start = np.meshgrid(x_starts, y_starts)
        start_points = np.column_stack([
            X_start.ravel(), 
            Y_start.ravel(),
            np.ones(len(X_start.ravel())) * 1000
        ])
    
    streamlines = []
    dt = 100  # Time step (s)
    
    for start in start_points:
        streamline = [start]
        point = start.copy()
        
        for _ in range(max_steps):
            # Interpolate velocity at current point
            try:
                u = interp_u([point[0], point[1]])[0]
                v = interp_v([point[0], point[1]])[0]
                
                if w_grid is not None and len(point) > 2:
                    # Simple vertical velocity (would need 3D interpolator for full 3D)
                    w = 0.1  # Placeholder
                else:
                    w = 0
            except:
                break
            
            # Check for NaN or stagnation
            if np.isnan(u) or np.isnan(v) or (abs(u) < 0.01 and abs(v) < 0.01):
                break
            
            # Update position (Runge-Kutta would be better)
            point = point + dt * np.array([u, v, w])
            
            # Check bounds
            if (point[0] < np.min(x) or point[0] > np.max(x) or
                point[1] < np.min(y) or point[1] > np.max(y)):
                break
            
            streamline.append(point.copy())
        
        if len(streamline) > 10:  # Only keep substantial streamlines
            streamlines.append(np.array(streamline))
    
    return streamlines
