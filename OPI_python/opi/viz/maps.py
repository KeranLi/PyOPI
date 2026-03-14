"""
Map visualization functions for OPI results.

Provides map-based visualization of topography, precipitation,
isotope distributions, and other OPI outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


def plot_topography_map(lon, lat, h_grid, title='Topography', 
                       cmap='terrain', figsize=(10, 8), save_path=None,
                       wind_arrows=None):
    """
    Create a map of topography.
    
    Parameters
    ----------
    lon, lat : ndarray
        Longitude and latitude grid vectors
    h_grid : ndarray
        Topography grid (m)
    title : str
        Map title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    wind_arrows : list of dict, optional
        List of wind arrows to plot. Each dict should have:
        'lon', 'lat', 'u', 'v', 'color' keys
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create shaded relief
    ls = LightSource(azdeg=315, altdeg=45)
    
    im = ax.pcolormesh(lon, lat, h_grid, cmap=cmap, shading='auto')
    
    # Add hillshade overlay
    rgb = ls.shade(h_grid, plt.cm.gray)
    ax.pcolormesh(lon, lat, rgb, shading='auto', alpha=0.3)
    
    # Plot wind arrows if provided
    if wind_arrows:
        for arrow in wind_arrows:
            ax.quiver(arrow['lon'], arrow['lat'], 
                     arrow['u'], arrow['v'],
                     color=arrow.get('color', 'blue'),
                     scale=50, width=0.005, headwidth=5)
    
    cbar = plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_precipitation_map(lon, lat, p_grid, title='Precipitation Rate',
                          cmap='Blues', figsize=(10, 8), save_path=None,
                          units=None, wind_arrow=None):
    """
    Create a map of precipitation rate.
    
    Parameters
    ----------
    lon, lat : ndarray
        Longitude and latitude grid vectors
    p_grid : ndarray
        Precipitation grid (kg/m^2/s or other units)
    title : str
        Map title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    units : str, optional
        Colorbar label (e.g., 'mm/day', 'Moisture Ratio', 'K')
    wind_arrow : dict, optional
        Wind arrow to plot with 'lon', 'lat', 'u', 'v', 'color' keys
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine units label and conversion
    if units is None:
        # Default: convert from kg/m^2/s to mm/hr (MATLAB convention)
        # 1 kg/m^2/s = 1 mm/s = 3600 mm/hr = 3.6e3 mm/hr
        display_grid = p_grid * 3.6e3
        cbar_label = 'Precipitation (mm/hr)'
        vmin, vmax = 0, 10
    elif units == 'Moisture Ratio':
        display_grid = p_grid
        cbar_label = 'Moisture Ratio'
        vmin, vmax = 0, 1
    elif units == 'Temperature (K)':
        display_grid = p_grid
        cbar_label = 'Temperature (K)'
        vmin, vmax = None, None
    elif units == "u'/U":
        display_grid = p_grid
        cbar_label = "u'/U"
        vmin, vmax = 0, 2
    else:
        display_grid = p_grid
        cbar_label = units
        vmin, vmax = None, None
    
    im = ax.pcolormesh(lon, lat, display_grid, cmap=cmap, 
                       shading='auto', vmin=vmin, vmax=vmax)
    
    # Plot wind arrow if provided
    if wind_arrow:
        ax.quiver(wind_arrow['lon'], wind_arrow['lat'],
                 wind_arrow['u'], wind_arrow['v'],
                 color=wind_arrow.get('color', 'black'),
                 scale=50, width=0.005, headwidth=5)
    
    cbar = plt.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_isotope_map(lon, lat, d2h_grid, d18o_grid=None, 
                    title_prefix='', cmap='RdYlBu_r', 
                    figsize=(14, 6), save_path=None):
    """
    Create maps of isotope distributions.
    
    Parameters
    ----------
    lon, lat : ndarray
        Longitude and latitude grid vectors
    d2h_grid : ndarray
        d2H grid (fraction or permil)
    d18o_grid : ndarray, optional
        d18O grid (fraction or permil)
    title_prefix : str
        Prefix for subplot titles
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib objects
    """
    if d18o_grid is not None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # d2H map
        im1 = axes[0].pcolormesh(lon, lat, d2h_grid * 1000, cmap=cmap, 
                                 shading='auto', vmin=-90, vmax=10)
        plt.colorbar(im1, ax=axes[0], label='d2H (permil)')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title(f'{title_prefix} d2H')
        
        # d18O map
        im2 = axes[1].pcolormesh(lon, lat, d18o_grid * 1000, cmap=cmap,
                                 shading='auto', vmin=-20, vmax=0)
        plt.colorbar(im2, ax=axes[1], label='d18O (permil)')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title(f'{title_prefix} d18O')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(lon, lat, d2h_grid * 1000, cmap=cmap,
                          shading='auto', vmin=-90, vmax=10)
        plt.colorbar(im, ax=ax, label='d2H (permil)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{title_prefix} d2H')
        axes = ax
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_precipitation_source_map(lon, lat, fraction_p_grid,
                                  map_limits=None,
                                  wind_arrows=None,
                                  title='Precipitation Source',
                                  cmap='coolwarm',
                                  figsize=(10, 8), save_path=None):
    """
    Create a map showing precipitation source fraction.
    
    Equivalent to MATLAB Fig09: shows fraction from state 1 (blue) vs state 2 (red).
    
    Parameters
    ----------
    lon, lat : ndarray
        Longitude and latitude grid vectors
    fraction_p_grid : ndarray
        Fraction from precipitation state 1 (0 = all state 2, 1 = all state 1)
    map_limits : list, optional
        [min_lon, max_lon, min_lat, max_lat] for axis limits
    wind_arrows : list of dict, optional
        List of wind arrows with 'lon', 'lat', 'u', 'v', 'color' keys
    title : str
        Map title
    cmap : str
        Colormap name (default 'coolwarm' for red-blue)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create precipitation source map
    im = ax.pcolormesh(lon, lat, fraction_p_grid, cmap=cmap,
                       shading='auto', vmin=0, vmax=1)
    
    # Plot wind arrows if provided
    if wind_arrows:
        for arrow in wind_arrows:
            ax.quiver(arrow['lon'], arrow['lat'],
                     arrow['u'], arrow['v'],
                     color=arrow.get('color', 'blue'),
                     scale=50, width=0.005, headwidth=5)
    
    cbar = plt.colorbar(im, ax=ax, label='Fraction, Source #1 (blue arrow)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    if map_limits:
        ax.set_xlim(map_limits[0], map_limits[1])
        ax.set_ylim(map_limits[2], map_limits[3])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_two_wind_comparison(data, save_dir=None):
    """
    Create comparison maps for two-wind OPI results.
    
    Parameters
    ----------
    data : dict
        Dictionary with two-wind OPI results
    save_dir : str, optional
        Directory to save figures
    
    Returns
    -------
    list
        List of created figures
    """
    figures = []
    
    lon = data['lon']
    lat = data['lat']
    
    # Combined topography (same for both states)
    if 'h_grid' in data:
        fig, _ = plot_topography_map(lon, lat, data['h_grid'],
                                     title='Topography - Two Wind Model')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_topography.png', dpi=300)
    
    # State 1 precipitation
    if 'p_grid_1' in data:
        fig, _ = plot_precipitation_map(lon, lat, data['p_grid_1'],
                                        title='Precipitation Rate - State #1')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_precipitation_state1.png', dpi=300)
    
    # State 2 precipitation
    if 'p_grid_2' in data:
        fig, _ = plot_precipitation_map(lon, lat, data['p_grid_2'],
                                        title='Precipitation Rate - State #2')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_precipitation_state2.png', dpi=300)
    
    # Combined precipitation
    if 'p_grid_1' in data and 'p_grid_2' in data:
        p_combined = data['p_grid_1'] + data['p_grid_2']
        fig, _ = plot_precipitation_map(lon, lat, p_combined,
                                        title='Precipitation Rate - Combined')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_precipitation_combined.png', dpi=300)
    
    # State 1 isotopes
    if 'd2h_grid_1' in data:
        fig, _ = plot_isotope_map(lon, lat, data['d2h_grid_1'],
                                  title_prefix='State #1')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_isotopes_state1.png', dpi=300)
    
    # State 2 isotopes
    if 'd2h_grid_2' in data:
        fig, _ = plot_isotope_map(lon, lat, data['d2h_grid_2'],
                                  title_prefix='State #2')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_isotopes_state2.png', dpi=300)
    
    return figures


def plot_result_maps(result_dict, two_wind=False, save_dir=None):
    """
    Create a complete set of maps from OPI results.
    
    Parameters
    ----------
    result_dict : dict
        Dictionary with OPI results including lon, lat, h_grid, p_grid, etc.
    two_wind : bool
        If True, generate two-wind specific maps
    save_dir : str, optional
        Directory to save figures
    
    Returns
    -------
    list
        List of created figures
    """
    figures = []
    
    lon = result_dict['lon']
    lat = result_dict['lat']
    
    # Topography map
    if 'h_grid' in result_dict:
        wind_arrows = None
        if two_wind and 'wind_1' in result_dict and 'wind_2' in result_dict:
            # Create wind arrows for both states
            wind_arrows = [
                {'lon': lon.min() + 0.1*(lon.max()-lon.min()),
                 'lat': lat.min() + 0.1*(lat.max()-lat.min()),
                 'u': 5, 'v': 5, 'color': 'blue'},
                {'lon': lon.max() - 0.1*(lon.max()-lon.min()),
                 'lat': lat.max() - 0.1*(lat.max()-lat.min()),
                 'u': -3, 'v': 7, 'color': 'red'}
            ]
        
        fig, _ = plot_topography_map(lon, lat, result_dict['h_grid'],
                                     title='Topography',
                                     wind_arrows=wind_arrows)
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_topography.png', dpi=300)
    
    # Precipitation map (combined or single)
    if 'p_grid' in result_dict:
        fig, _ = plot_precipitation_map(lon, lat, result_dict['p_grid'],
                                        title='Precipitation Rate')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_precipitation.png', dpi=300)
    
    # Two-wind specific maps
    if two_wind:
        # Precipitation source map (fraction from state 1)
        if 'fraction_p_grid' in result_dict:
            fig, _ = plot_precipitation_source_map(
                lon, lat, result_dict['fraction_p_grid'],
                wind_arrows=[
                    {'lon': lon.min() + 0.1*(lon.max()-lon.min()),
                     'lat': lat.min() + 0.1*(lat.max()-lat.min()),
                     'u': 5, 'v': 5, 'color': 'blue'},
                    {'lon': lon.max() - 0.1*(lon.max()-lon.min()),
                     'lat': lat.max() - 0.1*(lat.max()-lat.min()),
                     'u': -3, 'v': 7, 'color': 'red'}
                ]
            )
            figures.append(fig)
            if save_dir:
                fig.savefig(f'{save_dir}/map_precipitation_source.png', dpi=300)
        
        # Individual state precipitation
        if 'p_grid_1' in result_dict:
            fig, _ = plot_precipitation_map(lon, lat, result_dict['p_grid_1'],
                                            title='Precipitation Rate - State #1')
            figures.append(fig)
        
        if 'p_grid_2' in result_dict:
            fig, _ = plot_precipitation_map(lon, lat, result_dict['p_grid_2'],
                                            title='Precipitation Rate - State #2')
            figures.append(fig)
        
        # Individual state isotopes
        if 'd2h_grid_1' in result_dict:
            fig, _ = plot_isotope_map(lon, lat, result_dict['d2h_grid_1'],
                                      title_prefix='State #1')
            figures.append(fig)
        
        if 'd2h_grid_2' in result_dict:
            fig, _ = plot_isotope_map(lon, lat, result_dict['d2h_grid_2'],
                                      title_prefix='State #2')
            figures.append(fig)
    
    # Isotope maps (combined or single)
    if 'd2h_grid' in result_dict:
        d18o = result_dict.get('d18o_grid')
        fig, _ = plot_isotope_map(lon, lat, result_dict['d2h_grid'], d18o,
                                  title_prefix='Predicted')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_isotopes.png', dpi=300)
    
    return figures
