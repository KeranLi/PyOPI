"""
Map visualization functions for OPI results.

Provides map-based visualization of topography, precipitation,
isotope distributions, and other OPI outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource


def plot_topography_map(lon, lat, h_grid, title='Topography', 
                       cmap='terrain', figsize=(10, 8), save_path=None):
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
    
    cbar = plt.colorbar(im, ax=ax, label='Elevation (m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_precipitation_map(lon, lat, p_grid, title='Precipitation Rate',
                          cmap='Blues', figsize=(10, 8), save_path=None):
    """
    Create a map of precipitation rate.
    
    Parameters
    ----------
    lon, lat : ndarray
        Longitude and latitude grid vectors
    p_grid : ndarray
        Precipitation grid (mm/day)
    title : str
        Map title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.pcolormesh(lon, lat, p_grid * 1000 * 86400, cmap=cmap, 
                       shading='auto')  # Convert m/s to mm/day
    
    cbar = plt.colorbar(im, ax=ax, label='Precipitation (mm/day)')
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
                                 shading='auto', vmin=-150, vmax=-50)
        plt.colorbar(im1, ax=axes[0], label='d2H (permil)')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title(f'{title_prefix} d2H')
        
        # d18O map
        im2 = axes[1].pcolormesh(lon, lat, d18o_grid * 1000, cmap=cmap,
                                 shading='auto', vmin=-20, vmax=-5)
        plt.colorbar(im2, ax=axes[1], label='d18O (permil)')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title(f'{title_prefix} d18O')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.pcolormesh(lon, lat, d2h_grid * 1000, cmap=cmap,
                          shading='auto', vmin=-150, vmax=-50)
        plt.colorbar(im, ax=ax, label='d2H (permil)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{title_prefix} d2H')
        axes = ax
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_result_maps(result_dict, save_dir=None):
    """
    Create a complete set of maps from OPI results.
    
    Parameters
    ----------
    result_dict : dict
        Dictionary with OPI results including lon, lat, h_grid, p_grid, etc.
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
        fig, _ = plot_topography_map(lon, lat, result_dict['h_grid'],
                                     title='Topography')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_topography.png', dpi=300)
    
    # Precipitation map
    if 'p_grid' in result_dict:
        fig, _ = plot_precipitation_map(lon, lat, result_dict['p_grid'],
                                        title='Precipitation Rate')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_precipitation.png', dpi=300)
    
    # Isotope maps
    if 'd2h_grid' in result_dict:
        d18o = result_dict.get('d18o_grid')
        fig, _ = plot_isotope_map(lon, lat, result_dict['d2h_grid'], d18o,
                                  title_prefix='Predicted')
        figures.append(fig)
        if save_dir:
            fig.savefig(f'{save_dir}/map_isotopes.png', dpi=300)
    
    return figures
