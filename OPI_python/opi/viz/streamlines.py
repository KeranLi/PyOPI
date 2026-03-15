"""
3D Streamline visualization for OPI model

Generates 3D streamline plots (Fig 03 and Fig 04 in MATLAB)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_streamlines(lon, lat, h_grid, section_lon0, section_lat0, section_h0,
                        wind_azimuth, wind_speed, 
                        lon_arrow_start, lat_arrow_start, 
                        lon_arrow_offset, lat_arrow_offset,
                        title='3D Streamlines', save_path=None):
    """
    Create 3D streamline plot over topography.
    
    Matches MATLAB's Figure 3/4 style with proper terrain coloring.
    
    Parameters
    ----------
    lon, lat : 1D array
        Longitude and latitude coordinate vectors
    h_grid : 2D array
        Topography elevation (m), shape (len(lat), len(lon))
    section_lon0, section_lat0, section_h0 : float
        Section origin coordinates
    wind_azimuth : float
        Wind direction (degrees from North)
    wind_speed : float
        Wind speed (m/s)
    lon_arrow_start, lat_arrow_start : float
        Arrow start position for wind direction
    lon_arrow_offset, lat_arrow_offset : float
        Arrow direction offsets
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    from matplotlib.colors import LightSource
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ensure lon and lat are 1D arrays
    lon = np.asarray(lon).flatten()
    lat = np.asarray(lat).flatten()
    
    # Create meshgrid for surface plotting
    LON, LAT = np.meshgrid(lon, lat)
    
    # Scale elevation to km for better visualization
    h_km = h_grid * 1e-3
    section_h0_km = section_h0 * 1e-3
    
    # Create colormap similar to MATLAB's haxby
    # Use terrain colormap as base
    from .colormaps import haxby
    cmap = haxby(256)
    
    # Create surface plot with interpolated shading (similar to MATLAB's shading interp)
    surf = ax.plot_surface(LON, LAT, h_km, cmap='terrain', 
                          linewidth=0, antialiased=True,
                          rstride=1, cstride=1,
                          vmin=np.nanmin(h_km[h_km > 0]) if np.any(h_km > 0) else np.nanmin(h_km),
                          vmax=np.nanmax(h_km))
    
    # Plot section origin (white square with black border)
    ax.scatter([section_lon0], [section_lat0], [section_h0_km], 
              marker='s', s=300, c='white', edgecolors='black', linewidth=2.5)
    ax.scatter([section_lon0], [section_lat0], [section_h0_km], 
              marker='s', s=150, c='black', edgecolors='black', linewidth=1)
    
    # Plot wind direction arrow
    # MATLAB places arrow at max streamline height
    max_height = np.nanmax(h_km) * 1.5
    
    # Calculate proper arrow length in data coordinates
    lon_range = np.max(lon) - np.min(lon)
    lat_range = np.max(lat) - np.min(lat)
    
    # Use a reasonable arrow length (about 10% of domain)
    arrow_length = min(lon_range, lat_range) * 0.15
    
    # Convert azimuth to radians
    az_rad = np.deg2rad(wind_azimuth)
    
    # Arrow components
    dx = np.sin(az_rad) * arrow_length
    dy = np.cos(az_rad) * arrow_length
    
    # MATLAB places arrow at upper left area
    ax.quiver(lon_arrow_start, lat_arrow_start, max_height,
              dx, dy, 0,
              color='blue', linewidth=2.5, arrow_length_ratio=0.2)
    
    # Set view angle to match MATLAB (from left/behind looking downwind)
    # MATLAB uses azimuth relative to wind direction
    view_az = (60 - wind_azimuth) % 360 - 180
    ax.view_init(elev=30, azim=view_az)
    
    # Labels and formatting
    ax.set_xlabel('Longitude (deg)', fontsize=12)
    ax.set_ylabel('Latitude (deg)', fontsize=12)
    ax.set_zlabel('z (km)', fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Set axis limits
    ax.set_xlim([np.min(lon), np.max(lon)])
    ax.set_ylim([np.min(lat), np.max(lat)])
    ax.set_zlim([0, max_height * 1.2])
    
    # Set z-axis to show 0 at bottom
    ax.set_zticks([0, np.nanmax(h_km) * 0.5, np.nanmax(h_km)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def calculate_wind_arrow(grid_limits, azimuth, scale=0.1):
    """
    Calculate wind arrow position and direction for map plots.
    
    Parameters
    ----------
    grid_limits : dict
        {'lon_min', 'lon_max', 'lat_min', 'lat_max'}
    azimuth : float
        Wind direction (degrees from North)
    scale : float
        Arrow length scale
    
    Returns
    -------
    dict with arrow positions
    """
    # Center of grid
    lon_center = (grid_limits['lon_min'] + grid_limits['lon_max']) / 2
    lat_center = (grid_limits['lat_min'] + grid_limits['lat_max']) / 2
    
    # Arrow starts near the center
    lon_start = lon_center
    lat_start = lat_center
    
    # Convert azimuth to radians
    az_rad = np.deg2rad(azimuth)
    
    # Arrow direction (u=east, v=north)
    u = np.sin(az_rad) * scale
    v = np.cos(az_rad) * scale
    
    return {
        'lon_start': lon_start,
        'lat_start': lat_start,
        'lon_offset': u,
        'lat_offset': v
    }
