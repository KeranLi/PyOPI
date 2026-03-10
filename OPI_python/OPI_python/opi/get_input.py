"""
Input Data Loading Module

This module handles loading of topography and sample data for OPI calculations.
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from .coordinates import lonlat2xy
from .constants import RADIUS_EARTH


def grid_read(mat_file_path):
    """
    Extract grid variables from a MATLAB .mat file.
    
    The function looks for and extracts an mxn numerical array
    and two matching vectors with lengths m and n, which correspond
    to the y and x grid vectors.
    
    Parameters
    ----------
    mat_file_path : str
        Path to the MATLAB .mat file
    
    Returns
    -------
    x, y, z : ndarray
        Grid vectors x, y and the grid z
    """
    try:
        mat_data = loadmat(mat_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"MAT file not found: {mat_file_path}")
    except Exception as e:
        raise ValueError(f"Error loading MAT file: {e}")
    
    # Find numeric arrays in the file
    numeric_vars = {}
    for key, value in mat_data.items():
        # Skip metadata
        if key.startswith('__'):
            continue
        if isinstance(value, np.ndarray):
            numeric_vars[key] = value
    
    if len(numeric_vars) < 3:
        raise ValueError("MAT file must contain at least 3 numeric variables")
    
    # Find the grid (largest 2D array)
    grid_key = None
    grid_size = 0
    for key, value in numeric_vars.items():
        if value.ndim >= 2:
            size = value.shape[0] * value.shape[1]
            if size > grid_size:
                grid_size = size
                grid_key = key
    
    if grid_key is None:
        raise ValueError("No 2D grid found in MAT file")
    
    z = numeric_vars[grid_key]
    n_y, n_x = z.shape
    
    # Find x and y vectors
    x_key = None
    y_key = None
    
    for key, value in numeric_vars.items():
        if key == grid_key:
            continue
        if value.ndim == 2 and value.shape[0] == 1:
            value = value.flatten()
        if value.ndim == 1:
            if len(value) == n_x and x_key is None:
                x_key = key
            elif len(value) == n_y and y_key is None:
                y_key = key
    
    if x_key is None or y_key is None:
        raise ValueError(f"Could not find matching grid vectors. Grid shape: {z.shape}")
    
    x = numeric_vars[x_key].flatten()
    y = numeric_vars[y_key].flatten()
    
    return x, y, z


def tukey_window(n, r_tukey):
    """
    Create a Tukey window.
    
    Parameters
    ----------
    n : int
        Window length
    r_tukey : float
        Fractional width of cosine-taper section on each margin
    
    Returns
    -------
    window : ndarray
        Tukey window of length n
    """
    window = np.ones(n)
    if r_tukey <= 0:
        return window
    
    # Calculate taper width
    taper_width = int(np.floor(r_tukey / 2 * n))
    if taper_width < 1:
        return window
    
    # Create cosine taper
    taper = 0.5 * (1 - np.cos(np.pi * np.arange(taper_width) / taper_width))
    
    # Apply taper to both ends
    window[:taper_width] = taper
    window[-taper_width:] = taper[::-1]
    
    return window


def estimate_mwl(d18o, d2h, sd_res_ratio=28.3):
    """
    Estimate meteoric water line (MWL) using total least squares.
    
    Parameters
    ----------
    d18o, d2h : ndarray
        Isotope data (fraction, not permil)
    sd_res_ratio : float
        Estimated standard-deviation ratio for isotopic variation
    
    Returns
    -------
    b_hat : ndarray
        Coefficients for best fit: d2H = b_hat[0] + b_hat[1] * d18O
    sd_min, sd_max : float
        Minimum and maximum standard deviation
    cov : ndarray
        Estimated covariance matrix for isotopes
    i_fit : ndarray
        Logical vector indicating samples used for fit
    """
    # Identify samples with dExcess > 5 permil
    d_excess = d2h - 8 * d18o
    i_fit = d_excess > 5e-3
    n_fit = np.sum(i_fit)
    
    if n_fit < 2:
        # Not enough samples, use global MWL
        b_hat = np.array([9.47e-3, 8.03])
        sd_min = 1e-3
        sd_max = sd_min * sd_res_ratio
        cov = np.array([[sd_max**2, 0], [0, sd_min**2]])
        return b_hat, sd_min, sd_max, cov, i_fit
    
    # Total least squares using eigen decomposition
    d2h_mean = np.mean(d2h[i_fit])
    d18o_mean = np.mean(d18o[i_fit])
    
    A = np.column_stack([d18o[i_fit] - d18o_mean, d2h[i_fit] - d2h_mean])
    
    # Eigen decomposition of covariance matrix
    eigvals, eigvecs = np.linalg.eig(A.T @ A / (n_fit - 1))
    
    # Sort by eigenvalue
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    sd_min = np.sqrt(eigvals[0])
    sd_max = np.sqrt(eigvals[1])
    
    # Slope and intercept
    # d2H - d2H_mean = slope * (d18O - d18O_mean)
    slope = eigvecs[1, 1] / eigvecs[0, 1]
    intercept = d2h_mean - slope * d18o_mean
    b_hat = np.array([intercept, slope])
    
    # Create covariance matrix
    D_diag = np.array([eigvals[0], sd_res_ratio**2 * eigvals[0]])
    cov_temp = eigvecs @ np.diag(D_diag) @ eigvecs.T
    
    # Reorganize so V(d2H) and V(d18O) are 1st and 2nd on diagonal
    cov = np.array([[cov_temp[1, 1], cov_temp[0, 1]],
                    [cov_temp[1, 0], cov_temp[0, 0]]])
    
    return b_hat, sd_min, sd_max, cov, i_fit


def get_input(data_path, topo_file, r_tukey=0.0, sample_file=None, sd_res_ratio=28.3):
    """
    Load and parse input data for OPI calculations.
    
    Parameters
    ----------
    data_path : str
        Path to data directory
    topo_file : str
        Name of topography MAT file
    r_tukey : float, optional
        Fractional width of cosine-taper window. Default is 0.0.
    sample_file : str, optional
        Name of sample Excel file. Default is None.
    sd_res_ratio : float, optional
        Standard deviation ratio for MWL estimation. Default is 28.3.
    
    Returns
    -------
    dict : Dictionary containing all input data
    """
    import os
    
    # Global MWL (from GNIP data)
    b_mwl_sample = np.array([9.47e-3, 8.03])
    
    # Read topographic data
    topo_path = os.path.join(data_path, topo_file)
    lon, lat, h_grid = grid_read(topo_path)
    
    # Check for NaN values
    if np.any(np.isnan(h_grid)):
        raise ValueError("Elevation grid contains NaN values")
    
    # Set elevations below sea level to 0
    h_grid[h_grid < 0] = 0
    
    # Apply Tukey window if requested
    if r_tukey > 0:
        n_y, n_x = h_grid.shape
        window_y = tukey_window(n_y, r_tukey)
        window_x = tukey_window(n_x, r_tukey)
        window = np.outer(window_y, window_x)
        h_grid = window * h_grid
    
    # Initialize sample variables
    sample_line = np.array([])
    sample_lon = np.array([])
    sample_lat = np.array([])
    sample_x = np.array([])
    sample_y = np.array([])
    sample_d2h = np.array([])
    sample_d18o = np.array([])
    sample_d_excess = np.array([])
    sample_lc = np.array([])
    
    sample_line_alt = np.array([])
    sample_lon_alt = np.array([])
    sample_lat_alt = np.array([])
    sample_x_alt = np.array([])
    sample_y_alt = np.array([])
    sample_d2h_alt = np.array([])
    sample_d18o_alt = np.array([])
    sample_d_excess_alt = np.array([])
    sample_lc_alt = np.array([])
    
    sd_data_min = None
    sd_data_max = None
    cov = None
    
    # Read sample file if provided
    if sample_file is not None and sample_file.lower() != 'no':
        sample_path = os.path.join(data_path, sample_file)
        
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        
        # Read Excel file
        try:
            df = pd.read_excel(sample_path, header=None, comment='%')
        except Exception as e:
            raise ValueError(f"Error reading sample file: {e}")
        
        # Filter out comment rows
        df = df[~df.iloc[:, 0].astype(str).str.strip().str.startswith('%')]
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Line numbers (1-indexed for reference)
        sample_line = df.index.values + 1
        
        # Parse sample data
        sample_lon = df.iloc[:, 0].values
        sample_lat = df.iloc[:, 1].values
        sample_d2h = df.iloc[:, 2].values * 1e-3  # Convert permil to fraction
        sample_d18o = df.iloc[:, 3].values * 1e-3
        sample_lc = df.iloc[:, 4].astype(str).str.upper().str[0].values
        
        # Check that samples lie within grid
        if np.any((lon[0] > sample_lon) | (lon[-1] < sample_lon) |
                  (lat[0] > sample_lat) | (lat[-1] < sample_lat)):
            i_problem = (lon[0] > sample_lon) | (lon[-1] < sample_lon) | \
                       (lat[0] > sample_lat) | (lat[-1] < sample_lat)
            print("Warning: Some samples lie outside the topographic grid")
            for i in np.where(i_problem)[0]:
                print(f"  Line {sample_line[i]}: {sample_lon[i]}, {sample_lat[i]}")
        
        # Check sample type
        valid_lc = np.isin(sample_lc, ['L', 'C'])
        if not np.all(valid_lc):
            print("Warning: Type variable can only have values of L or C")
        
        # Estimate MWL
        b_mwl_sample, sd_data_min, sd_data_max, cov, i_fit = \
            estimate_mwl(sample_d18o, sample_d2h, sd_res_ratio)
        
        # Divide samples into primary and altered precipitation
        sample_line_alt = sample_line[~i_fit]
        sample_lon_alt = sample_lon[~i_fit]
        sample_lat_alt = sample_lat[~i_fit]
        sample_d2h_alt = sample_d2h[~i_fit]
        sample_d18o_alt = sample_d18o[~i_fit]
        sample_lc_alt = sample_lc[~i_fit]
        
        sample_line = sample_line[i_fit]
        sample_lon = sample_lon[i_fit]
        sample_lat = sample_lat[i_fit]
        sample_d2h = sample_d2h[i_fit]
        sample_d18o = sample_d18o[i_fit]
        sample_lc = sample_lc[i_fit]
        
        # Calculate d-excess
        sample_d_excess = sample_d2h - 8 * sample_d18o
        sample_d_excess_alt = sample_d2h_alt - 8 * sample_d18o_alt
    
    # Define map origin
    if len(sample_lon) > 0:
        lon0 = np.mean(sample_lon)
        lat0 = np.mean(sample_lat)
        
        # Convert samples to projected coordinates
        sample_x, sample_y = lonlat2xy(sample_lon, sample_lat, lon0, lat0)
        if len(sample_lon_alt) > 0:
            sample_x_alt, sample_y_alt = lonlat2xy(sample_lon_alt, sample_lat_alt, lon0, lat0)
    else:
        lon0 = np.mean(lon)
        lat0 = np.mean(lat)
        sample_x_alt = np.array([])
        sample_y_alt = np.array([])
    
    # Convert grid to projected coordinates
    x, y = lonlat2xy(lon, lat, lon0, lat0)
    
    # Coriolis frequency (rad/s)
    omega = 7.2921e-5  # Earth's rotation rate
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    # Calculate h_r (characteristic distance for isotope exchange)
    h_r = 540.0  # m (typical value)
    
    return {
        'lon': lon,
        'lat': lat,
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'lon0': lon0,
        'lat0': lat0,
        'sample_line': sample_line,
        'sample_lon': sample_lon,
        'sample_lat': sample_lat,
        'sample_x': sample_x,
        'sample_y': sample_y,
        'sample_d2h': sample_d2h,
        'sample_d18o': sample_d18o,
        'sample_d_excess': sample_d_excess,
        'sample_lc': sample_lc,
        'sample_line_alt': sample_line_alt,
        'sample_lon_alt': sample_lon_alt,
        'sample_lat_alt': sample_lat_alt,
        'sample_x_alt': sample_x_alt,
        'sample_y_alt': sample_y_alt,
        'sample_d2h_alt': sample_d2h_alt,
        'sample_d18o_alt': sample_d18o_alt,
        'sample_d_excess_alt': sample_d_excess_alt,
        'sample_lc_alt': sample_lc_alt,
        'b_mwl_sample': b_mwl_sample,
        'sd_data_min': sd_data_min,
        'sd_data_max': sd_data_max,
        'cov': cov,
        'f_c': f_c,
        'h_r': h_r
    }


if __name__ == "__main__":
    print("Testing get_input module...")
    
    # Test Tukey window
    print("\nTest 1: Tukey window")
    win = tukey_window(100, 0.2)
    print(f"  Window shape: {win.shape}")
    print(f"  Window range: {win.min():.3f} to {win.max():.3f}")
    
    # Test MWL estimation
    print("\nTest 2: MWL estimation")
    d18o_test = np.array([-10, -8, -6, -4, -2]) * 1e-3
    d2h_test = np.array([-70, -55, -40, -25, -10]) * 1e-3
    b_hat, sd_min, sd_max, cov, i_fit = estimate_mwl(d18o_test, d2h_test)
    print(f"  MWL: d2H = {b_hat[0]*1000:.2f} + {b_hat[1]:.2f} * d18O")
    print(f"  sd_min: {sd_min*1000:.2f} permil")
    print(f"  sd_max: {sd_max*1000:.2f} permil")
    
    print("\nTests completed!")
