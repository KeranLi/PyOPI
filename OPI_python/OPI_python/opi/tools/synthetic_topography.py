"""
Synthetic Topography Generation

Generates synthetic digital elevation models (DEMs) for testing OPI.
Ported from MATLAB opiSyntheticTopography.m

Includes:
- Gaussian mountain ranges
- Sinusoidal topography
- Ridge topography
"""

import numpy as np
from scipy.io import savemat


def gaussian_topography(x, y, amplitude=3000, sigma=None, center=None):
    """
    Create Gaussian-shaped topography.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors (m)
    amplitude : float
        Maximum height (m)
    sigma : float or tuple
        Gaussian width(s) in x and y directions (m)
    center : tuple
        Center position (x0, y0) in meters
    
    Returns
    -------
    h_grid : ndarray
        Topography grid (m)
    """
    X, Y = np.meshgrid(x, y)
    
    if sigma is None:
        sigma = (np.max(x) - np.min(x)) / 6, (np.max(y) - np.min(y)) / 6
    
    if isinstance(sigma, (int, float)):
        sigma_x = sigma_y = sigma
    else:
        sigma_x, sigma_y = sigma
    
    if center is None:
        x0, y0 = 0, 0
    else:
        x0, y0 = center
    
    h_grid = amplitude * np.exp(-((X - x0)**2 / (2 * sigma_x**2) + 
                                  (Y - y0)**2 / (2 * sigma_y**2)))
    
    return h_grid


def sinusoidal_topography(x, y, amplitude=3000, wavelength=100e3, 
                          direction='east', offset_horizontal=100e3,
                          offset_vertical=1):
    """
    Create sinusoidal topography.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors (m)
    amplitude : float
        Wave amplitude (trough to crest) in meters
    wavelength : float
        Wavelength in meters
    direction : str
        'east' or 'north' - direction of wave propagation
    offset_horizontal : float
        Horizontal offset from grid edge (m)
    offset_vertical : float
        Vertical offset (m) - minimum elevation
    
    Returns
    -------
    h_grid : ndarray
        Topography grid (m)
    """
    x_min, y_min = np.min(x), np.min(y)
    n_y, n_x = len(y), len(x)
    
    if direction == 'north':
        # North-directed sinusoids
        h_profile = amplitude * (1 + np.sin(2 * np.pi * 
            (y - (y_min + offset_horizontal)) / wavelength - np.pi/2)) / 2
        h_profile[y < (y_min + offset_horizontal)] = 0
        h_grid = np.tile(h_profile[:, np.newaxis], (1, n_x))
    else:
        # East-directed sinusoids (default)
        h_profile = amplitude * (1 + np.sin(2 * np.pi * 
            (x - (x_min + offset_horizontal)) / wavelength - np.pi/2)) / 2
        h_profile[x < (x_min + offset_horizontal)] = 0
        h_grid = np.tile(h_profile[np.newaxis, :], (n_y, 1))
    
    h_grid = h_grid + offset_vertical
    
    return h_grid


def ridge_topography(x, y, amplitude=3000, width=50e3, 
                     direction='east', center=None):
    """
    Create ridge-shaped topography.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors (m)
    amplitude : float
        Ridge height (m)
    width : float
        Ridge width at half-height (m)
    direction : str
        'east' or 'north' - ridge orientation
    center : float
        Center position in perpendicular direction (m)
    
    Returns
    -------
    h_grid : ndarray
        Topography grid (m)
    """
    X, Y = np.meshgrid(x, y)
    
    if center is None:
        center = 0
    
    if direction == 'north':
        # North-south ridge (varying in x)
        distance = np.abs(X - center)
    else:
        # East-west ridge (varying in y)
        distance = np.abs(Y - center)
    
    # Gaussian-shaped ridge
    h_grid = amplitude * np.exp(-(distance**2) / (2 * (width/2.355)**2))
    
    return h_grid


def create_synthetic_dem(topo_type='gaussian', grid_size=(700e3, 700e3),
                        grid_spacing=(1000, 1000), lon0=0, lat0=45,
                        amplitude=3000, output_file=None, **kwargs):
    """
    Create synthetic DEM and save to MAT file.
    
    Parameters
    ----------
    topo_type : str
        'gaussian', 'sinusoidal_east', 'sinusoidal_north', or 'ridge'
    grid_size : tuple
        (size_x, size_y) in meters
    grid_spacing : tuple
        (dx, dy) in meters
    lon0, lat0 : float
        Center longitude and latitude (degrees)
    amplitude : float
        Maximum topography height (m)
    output_file : str
        Output MAT file path
    **kwargs : dict
        Additional parameters for specific topography types
    
    Returns
    -------
    dem : dict
        Dictionary with 'lon', 'lat', 'hGrid', 'x', 'y'
    """
    from ..io.coordinates import xy2lonlat
    
    size_x, size_y = grid_size
    dx, dy = grid_spacing
    
    # Create grid vectors
    n_x = int(np.ceil(size_x / dx)) + 1
    dx = size_x / (n_x - 1)
    x_min = -size_x / 2
    x = x_min + np.arange(n_x) * dx
    
    n_y = int(np.ceil(size_y / dy)) + 1
    dy = size_y / (n_y - 1)
    y_min = -size_y / 2
    y = y_min + np.arange(n_y) * dy
    
    # Generate topography
    if topo_type == 'gaussian':
        sigma = kwargs.get('sigma', size_x / 6)
        center = kwargs.get('center', (0, 0))
        h_grid = gaussian_topography(x, y, amplitude, sigma, center)
    
    elif topo_type == 'sinusoidal_east':
        wavelength = kwargs.get('wavelength', 100e3)
        offset_h = kwargs.get('offset_horizontal', 100e3)
        offset_v = kwargs.get('offset_vertical', 1)
        h_grid = sinusoidal_topography(x, y, amplitude, wavelength, 
                                       'east', offset_h, offset_v)
    
    elif topo_type == 'sinusoidal_north':
        wavelength = kwargs.get('wavelength', 100e3)
        offset_h = kwargs.get('offset_horizontal', 100e3)
        offset_v = kwargs.get('offset_vertical', 1)
        h_grid = sinusoidal_topography(x, y, amplitude, wavelength,
                                       'north', offset_h, offset_v)
    
    elif topo_type == 'ridge':
        width = kwargs.get('width', 50e3)
        direction = kwargs.get('direction', 'east')
        center = kwargs.get('center', 0)
        h_grid = ridge_topography(x, y, amplitude, width, direction, center)
    
    else:
        raise ValueError(f"Unknown topography type: {topo_type}")
    
    # Convert to lon/lat
    lon, lat = xy2lonlat(x, y, lon0, lat0)
    
    # Format for output (lon as row, lat as column)
    lon = lon.reshape(1, -1)
    lat = lat.reshape(-1, 1)
    
    dem = {
        'lon': lon,
        'lat': lat,
        'hGrid': h_grid,
        'x': x,
        'y': y,
        'lon0': lon0,
        'lat0': lat0
    }
    
    # Save to file if requested
    if output_file:
        savemat(output_file, dem)
        print(f"Saved synthetic DEM to: {output_file}")
    
    return dem


if __name__ == "__main__":
    # Create example topographies
    
    # 1. Gaussian mountain at 45N
    print("Creating Gaussian topography...")
    dem_gaussian = create_synthetic_dem(
        topo_type='gaussian',
        grid_size=(500e3, 500e3),
        grid_spacing=(2000, 2000),
        lon0=0, lat0=45,
        amplitude=2000,
        sigma=(50e3, 50e3),
        output_file='gaussian_topography.mat'
    )
    print(f"Grid shape: {dem_gaussian['hGrid'].shape}")
    print(f"Max elevation: {np.max(dem_gaussian['hGrid']):.1f} m")
    
    # 2. East-directed sinusoids
    print("\nCreating east-directed sinusoidal topography...")
    dem_sin_east = create_synthetic_dem(
        topo_type='sinusoidal_east',
        grid_size=(700e3, 700e3),
        grid_spacing=(2000, 2000),
        lon0=0, lat0=0,
        amplitude=3000,
        wavelength=100e3,
        output_file='sinusoidal_east_topography.mat'
    )
    
    print("\nSynthetic topography generation complete!")
