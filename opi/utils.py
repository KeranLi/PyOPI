"""
Utility functions for OPI

This module provides various utility functions for data processing,
visualization helpers, and common calculations.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def save_grids_to_netcdf(x, y, grids_dict, filename, attributes=None):
    """
    Save grid data to NetCDF file.
    
    Parameters:
    -----------
    x, y : ndarray
        Grid coordinates
    grids_dict : dict
        Dictionary of grid arrays to save
    filename : str
        Output NetCDF file path
    attributes : dict, optional
        Global attributes for the file
    """
    try:
        import netCDF4
    except ImportError:
        raise ImportError("netCDF4 package required. Install with: pip install netCDF4")
    
    nc = netCDF4.Dataset(filename, 'w', format='NETCDF4')
    
    # Create dimensions
    nc.createDimension('x', len(x))
    nc.createDimension('y', len(y))
    
    # Create coordinate variables
    x_var = nc.createVariable('x', 'f8', ('x',))
    y_var = nc.createVariable('y', 'f8', ('y',))
    x_var[:] = x
    y_var[:] = y
    
    x_var.units = 'm'
    y_var.units = 'm'
    
    # Create grid variables
    for name, grid in grids_dict.items():
        if grid.shape == (len(y), len(x)):
            var = nc.createVariable(name, 'f8', ('y', 'x'))
            var[:] = grid
    
    # Add global attributes
    if attributes:
        for key, val in attributes.items():
            setattr(nc, key, val)
    
    nc.close()
    print(f"Saved grids to: {filename}")


def save_grids_to_numpy(x, y, grids_dict, filename):
    """
    Save grid data to NumPy .npz file.
    
    Parameters:
    -----------
    x, y : ndarray
        Grid coordinates
    grids_dict : dict
        Dictionary of grid arrays
    filename : str
        Output .npz file path
    """
    data = {'x': x, 'y': y}
    data.update(grids_dict)
    np.savez(filename, **data)
    print(f"Saved grids to: {filename}")


def interpolate_to_points(x, y, grid, points_x, points_y, method='linear'):
    """
    Interpolate grid values to specific points.
    
    Parameters:
    -----------
    x, y : ndarray
        Grid coordinates
    grid : ndarray
        2D grid array
    points_x, points_y : ndarray
        Points to interpolate to
    method : str
        Interpolation method ('linear', 'cubic', 'nearest')
    
    Returns:
    --------
    values : ndarray
        Interpolated values at points
    """
    interp = RegularGridInterpolator(
        (y, x), grid,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )
    
    points = np.column_stack([points_y, points_x])
    return interp(points)


def calculate_catchment_mean(x, y, grid, sample_x, sample_y, sample_ij_catch):
    """
    Calculate catchment-weighted mean for sample points.
    
    Parameters:
    -----------
    x, y : ndarray
        Grid coordinates
    grid : ndarray
        Grid to average (e.g., precipitation)
    sample_x, sample_y : ndarray
        Sample point coordinates
    sample_ij_catch : list of arrays
        Catchment indices for each sample
    
    Returns:
    --------
    means : ndarray
        Catchment-weighted mean for each sample
    """
    means = np.full(len(sample_x), np.nan)
    
    for i, ij in enumerate(sample_ij_catch):
        if len(ij) > 0:
            values = [grid[idx[0], idx[1]] for idx in ij]
            means[i] = np.mean(values)
    
    return means


def deuterium_excess(d2h, d18o):
    """
    Calculate deuterium excess.
    
    Parameters:
    -----------
    d2h : float or ndarray
        delta^2H values (fraction or permil)
    d18o : float or ndarray
        delta^18O values (fraction or permil)
    
    Returns:
    --------
    d_excess : float or ndarray
        Deuterium excess values
    """
    d2h = np.asarray(d2h)
    d18o = np.asarray(d18o)
    
    # Auto-detect if values are in permil or fraction
    if np.any(np.abs(d2h) > 0.1):  # Likely permil
        return d2h - 8 * d18o
    else:  # Fraction
        return (d2h - 8 * d18o) * 1000


def meteoric_water_line(d18o, slope=8.0, intercept=10.0):
    """
    Calculate expected d2H from d18O using meteoric water line.
    
    Parameters:
    -----------
    d18o : float or ndarray
        delta^18O values
    slope : float
        MWL slope (default: 8.0)
    intercept : float
        MWL intercept in permil (default: 10.0)
    
    Returns:
    --------
    d2h_expected : float or ndarray
        Expected d2H values
    """
    d18o = np.asarray(d18o)
    
    # Auto-detect units
    if np.any(np.abs(d18o) > 0.1):  # Permil
        return slope * d18o + intercept
    else:  # Fraction
        return (slope * d18o * 1000 + intercept) / 1000


def calculate_lapse_rate(elevation, temperature):
    """
    Calculate environmental lapse rate from elevation and temperature data.
    
    Parameters:
    -----------
    elevation : ndarray
        Elevation values (m)
    temperature : ndarray
        Temperature values (K)
    
    Returns:
    --------
    lapse_rate : float
        Environmental lapse rate (K/km)
    """
    # Simple linear regression
    coeffs = np.polyfit(elevation / 1000, temperature, 1)
    return -coeffs[0]  # K/km (negative means decreasing with height)


def wind_components(speed, azimuth):
    """
    Convert wind speed and azimuth to u, v components.
    
    Parameters:
    -----------
    speed : float or ndarray
        Wind speed (m/s)
    azimuth : float or ndarray
        Wind direction in degrees from North
    
    Returns:
    --------
    u, v : tuple
        Wind components (u=east, v=north)
    """
    azimuth_rad = np.deg2rad(azimuth)
    u = speed * np.sin(azimuth_rad)
    v = speed * np.cos(azimuth_rad)
    return u, v


def mountain_height_number(u, nm, h_max):
    """
    Calculate mountain-height number.
    
    M = N_m * h_max / U
    
    Parameters:
    -----------
    u : float
        Wind speed (m/s)
    nm : float
        Buoyancy frequency (rad/s)
    h_max : float
        Maximum mountain height (m)
    
    Returns:
    --------
    m : float
        Mountain-height number
    """
    return nm * h_max / u


def rossby_number(u, f, length_scale):
    """
    Calculate Rossby number.
    
    Ro = U / (f * L)
    
    Parameters:
    -----------
    u : float
        Wind speed (m/s)
    f : float
        Coriolis parameter (rad/s)
    length_scale : float
        Characteristic length scale (m)
    
    Returns:
    --------
    ro : float
        Rossby number
    """
    return u / (f * length_scale)


def froude_number(u, nm, h):
    """
    Calculate Froude number.
    
    Fr = U / (N_m * h)
    
    Parameters:
    -----------
    u : float
        Wind speed (m/s)
    nm : float
        Buoyancy frequency (rad/s)
    h : float
        Mountain height (m)
    
    Returns:
    --------
    fr : float
        Froude number
    """
    return u / (nm * h)


def summarize_results(result_dict):
    """
    Print a summary of calculation results.
    
    Parameters:
    -----------
    result_dict : dict
        Results dictionary from opi_calc_one_wind or opi_calc_two_winds
    """
    print("=" * 60)
    print("OPI Results Summary")
    print("=" * 60)
    
    if 'solution_params' in result_dict:
        print("\nSolution Parameters:")
        for name, val in result_dict['solution_params'].items():
            print(f"  {name}: {val}")
    
    if 'derived_params' in result_dict:
        print("\nDerived Parameters:")
        for name, val in result_dict['derived_params'].items():
            if isinstance(val, float):
                print(f"  {name}: {val:.6f}")
            else:
                print(f"  {name}: {val}")
    
    if 'results' in result_dict and result_dict['results']:
        print("\nOutput Grids:")
        for key in result_dict['results'].keys():
            if hasattr(result_dict['results'][key], 'shape'):
                print(f"  {key}: {result_dict['results'][key].shape}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test deuterium excess
    d2h = -100  # permil
    d18o = -12  # permil
    d_excess = deuterium_excess(d2h, d18o)
    print(f"d-excess for d2H={d2h}, d18O={d18o}: {d_excess:.1f}")
    
    # Test wind components
    u, v = wind_components(10.0, 90.0)
    print(f"Wind components (speed=10, azimuth=90): u={u:.2f}, v={v:.2f}")
    
    # Test dimensionless numbers
    ro = rossby_number(10.0, 1e-4, 100000)
    print(f"Rossby number: {ro:.4f}")
    
    fr = froude_number(10.0, 0.01, 2000)
    print(f"Froude number: {fr:.4f}")
    
    print("\nAll utility tests passed!")
