"""
Map Visualization Functions for OPI

This module provides 2D map visualization functions for orographic
precipitation and isotope model outputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, Union


def plot_topography_contour(x: np.ndarray,
                            y: np.ndarray,
                            h_grid: np.ndarray,
                            ax: Optional[plt.Axes] = None,
                            levels: Optional[Union[int, np.ndarray]] = None,
                            title: Optional[str] = None) -> plt.ContourSet:
    """
    Create a filled contour plot of topography.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    h_grid : array_like
        2D array of elevation values (m) with shape (len(y), len(x))
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    levels : int or array_like, optional
        Number of contour levels or specific level values. 
        If None, uses matplotlib's default levels.
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.contour.ContourSet
        The contour set object from filled contours
        
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    >>> cs = plot_topography_contour(x, y, h, title='Mountain Topography')
    
    >>> # Custom levels
    >>> cs = plot_topography_contour(x, y, h, levels=20, title='Detailed Topography')
    """
    x = np.asarray(x)
    y = np.asarray(y)
    h_grid = np.asarray(h_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(x, y)
    
    # Create filled contours
    if levels is not None:
        cs = ax.contourf(X, Y, h_grid, levels=levels, cmap='terrain', 
                         extend='neither')
    else:
        cs = ax.contourf(X, Y, h_grid, cmap='terrain', extend='neither')
    
    # Add contour lines
    ax.contour(X, Y, h_grid, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Elevation (m)')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    
    if title:
        ax.set_title(title)
    
    return cs


def plot_precipitation_contour(x: np.ndarray,
                               y: np.ndarray,
                               p_grid: np.ndarray,
                               ax: Optional[plt.Axes] = None,
                               levels: Optional[Union[int, np.ndarray]] = None,
                               title: Optional[str] = None) -> plt.ContourSet:
    """
    Create a filled contour plot of precipitation with custom colormap.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    p_grid : array_like
        2D array of precipitation values (mm/day or kg/m²/s) 
        with shape (len(y), len(x))
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    levels : int or array_like, optional
        Number of contour levels or specific level values.
        If None, uses default levels based on data range.
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.contour.ContourSet
        The contour set object from filled contours
        
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> p = 10 * np.exp(-((X - 5000)**2) / 2e6) * (1 + 0.5 * np.exp(-((Y - 5000)**2) / 1e6))
    >>> cs = plot_precipitation_contour(x, y, p, title='Orographic Precipitation')
    
    >>> # Custom levels
    >>> levels = np.linspace(0, 20, 21)
    >>> cs = plot_precipitation_contour(x, y, p, levels=levels)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    p_grid = np.asarray(p_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom precipitation colormap (white to blue)
    colors = ['#FFFFFF', '#E6F3FF', '#B3D9FF', '#80BFFF', '#4D94FF', 
              '#1A66FF', '#0040FF', '#0033CC', '#002699', '#001A66']
    precip_cmap = LinearSegmentedColormap.from_list('precipitation', colors)
    
    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(x, y)
    
    # Determine levels if not provided
    if levels is None:
        p_max = np.nanmax(p_grid)
        if p_max > 0:
            levels = np.linspace(0, p_max, 15)
        else:
            levels = 10
    
    # Create filled contours
    cs = ax.contourf(X, Y, p_grid, levels=levels, cmap=precip_cmap, 
                     extend='max')
    
    # Add contour lines
    ax.contour(X, Y, p_grid, levels=levels, colors='darkblue', 
               linewidths=0.5, alpha=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Precipitation (mm/day)')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    
    if title:
        ax.set_title(title)
    
    return cs


def plot_isotope_maps(x: np.ndarray,
                      y: np.ndarray,
                      d2H_grid: np.ndarray,
                      d18O_grid: np.ndarray,
                      figsize: Tuple[float, float] = (12, 5)) -> plt.Figure:
    """
    Create side-by-side maps of d2H and d18O isotope distributions.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    d2H_grid : array_like
        2D array of delta^2H values (fraction or permil)
    d18O_grid : array_like
        2D array of delta^18O values (fraction or permil)
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (12, 5).
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing both subplots
        
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> d2H = -100 + 20 * np.sin(X / 2000) * np.cos(Y / 2000)
    >>> d18O = -12 + 2.5 * np.sin(X / 2000) * np.cos(Y / 2000)
    >>> fig = plot_isotope_maps(x, y, d2H, d18O)
    
    >>> # Custom figure size
    >>> fig = plot_isotope_maps(x, y, d2H, d18O, figsize=(14, 6))
    """
    x = np.asarray(x)
    y = np.asarray(y)
    d2H_grid = np.asarray(d2H_grid)
    d18O_grid = np.asarray(d18O_grid)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Plot d2H
    vmax_2h = np.max(np.abs([d2H_grid.min(), d2H_grid.max()]))
    if vmax_2h > 1:  # Likely in permil
        units_2h = '‰'
        scale_2h = 1.0
    else:  # Likely in fraction
        units_2h = 'fraction'
        scale_2h = 1.0
    
    im1 = axes[0].contourf(X, Y, d2H_grid * scale_2h, levels=20, 
                           cmap='RdBu_r', vmin=-vmax_2h * scale_2h, 
                           vmax=vmax_2h * scale_2h)
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label(f'δ²H ({units_2h})')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Hydrogen Isotopes (δ²H)')
    axes[0].set_aspect('equal', adjustable='box')
    
    # Plot d18O
    vmax_18o = np.max(np.abs([d18O_grid.min(), d18O_grid.max()]))
    if vmax_18o > 1:  # Likely in permil
        units_18o = '‰'
        scale_18o = 1.0
    else:  # Likely in fraction
        units_18o = 'fraction'
        scale_18o = 1.0
    
    im2 = axes[1].contourf(X, Y, d18O_grid * scale_18o, levels=20, 
                           cmap='RdBu_r', vmin=-vmax_18o * scale_18o, 
                           vmax=vmax_18o * scale_18o)
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label(f'δ¹⁸O ({units_18o})')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Oxygen Isotopes (δ¹⁸O)')
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return fig


def plot_wind_field(x: np.ndarray,
                    y: np.ndarray,
                    U: np.ndarray,
                    azimuth: float,
                    h_grid: np.ndarray,
                    ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot streamlines of the wind field over topography.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    U : array_like
        Wind speed array. Can be scalar, 1D (x-varying), or 2D (x,y-varying).
    azimuth : float
        Wind direction in degrees from North (0 = North, 90 = East, etc.)
    h_grid : array_like
        2D array of elevation values (m) for background topography
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot
        
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    >>> U = 10.0  # m/s wind speed
    >>> azimuth = 270.0  # Westerly wind
    >>> fig = plot_wind_field(x, y, U, azimuth, h)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    h_grid = np.asarray(h_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Convert azimuth to mathematical angle (counter-clockwise from East)
    # Azimuth: 0 = North, 90 = East, etc.
    # Math angle: 0 = East, 90 = North, etc.
    wind_angle = np.radians(90 - azimuth)
    
    # Handle U as scalar, 1D, or 2D
    U = np.asarray(U)
    if U.ndim == 0:  # Scalar
        Ux = U * np.cos(wind_angle) * np.ones_like(X)
        Uy = U * np.sin(wind_angle) * np.ones_like(Y)
    elif U.ndim == 1:  # 1D array
        if len(U) == len(x):
            Ux = np.outer(np.ones(len(y)), U * np.cos(wind_angle))
            Uy = np.outer(np.ones(len(y)), U * np.sin(wind_angle))
        else:
            raise ValueError("U array length must match x array length")
    else:  # 2D array
        Ux = U * np.cos(wind_angle)
        Uy = U * np.sin(wind_angle)
    
    # Plot topography as background
    im = ax.contourf(X, Y, h_grid, levels=20, cmap='terrain', alpha=0.6)
    plt.colorbar(im, ax=ax, label='Elevation (m)')
    
    # Add topography contour lines
    ax.contour(X, Y, h_grid, levels=10, colors='brown', linewidths=0.5, alpha=0.7)
    
    # Plot streamlines
    speed = np.sqrt(Ux**2 + Uy**2)
    strm = ax.streamplot(X, Y, Ux, Uy, color=speed, cmap='viridis',
                         density=2, linewidth=1.5, arrowsize=1.5)
    plt.colorbar(strm.lines, ax=ax, label='Wind Speed (m/s)')
    
    # Mark wind direction
    ax.annotate('', xy=(0.85, 0.85), xytext=(0.85, 0.95),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(0.9, 0.9, f'{azimuth:.0f}°', transform=ax.transAxes,
            fontsize=12, color='red', fontweight='bold')
    
    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Wind Field (Azimuth: {azimuth:.1f}°, U: {np.mean(U):.1f} m/s)')
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def plot_moisture_ratio(x: np.ndarray,
                        y: np.ndarray,
                        f_m_grid: np.ndarray,
                        ax: Optional[plt.Axes] = None,
                        title: Optional[str] = None) -> plt.ContourSet:
    """
    Plot the moisture ratio field.
    
    The moisture ratio represents the fraction of original moisture remaining
    in the air parcel after precipitation. Values range from 1 (no precipitation)
    to near 0 (complete precipitation).
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    f_m_grid : array_like
        2D array of moisture ratio values (0 to 1)
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title. If None, uses default title.
        
    Returns
    -------
    matplotlib.contour.ContourSet
        The contour set object from filled contours
        
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> f_m = 1.0 - 0.5 * np.exp(-((X - 5000)**2) / 1e6)
    >>> cs = plot_moisture_ratio(x, y, f_m, title='Moisture Ratio Field')
    """
    x = np.asarray(x)
    y = np.asarray(y)
    f_m_grid = np.asarray(f_m_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid
    X, Y = np.meshgrid(x, y)
    
    # Create filled contours with custom colormap (blue to white to red)
    # Blue: high moisture (1.0), Red: low moisture (0.0)
    colors = ['#FF6666', '#FF9999', '#FFCCCC', '#FFFFFF', 
              '#CCE5FF', '#99CCFF', '#6699FF', '#3366FF', '#0033CC']
    moisture_cmap = LinearSegmentedColormap.from_list('moisture', colors)
    
    # Ensure proper bounds
    f_m_clipped = np.clip(f_m_grid, 0, 1)
    
    cs = ax.contourf(X, Y, f_m_clipped, levels=np.linspace(0, 1, 21),
                     cmap=moisture_cmap, vmin=0, vmax=1, extend='neither')
    
    # Add contour lines
    ax.contour(X, Y, f_m_clipped, levels=[0.2, 0.4, 0.6, 0.8],
               colors='gray', linewidths=0.5, alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('Moisture Ratio (f_m)')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Moisture Ratio Field')
    
    return cs


if __name__ == "__main__":
    # Test map visualization functions
    print("Testing map visualization functions...")
    
    # Create sample data
    x = np.linspace(0, 10000, 100)
    y = np.linspace(0, 10000, 100)
    X, Y = np.meshgrid(x, y)
    
    # Topography: Gaussian mountain
    h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    
    # Precipitation: orographic enhancement
    p = 10 * np.exp(-((X - 5000)**2) / 2e6) * (1 + 0.5 * h / 1000)
    
    # Isotopes: altitude and amount effect
    d2H = -100 + 0.02 * h + 2 * (p - 10)
    d18O = -12 + 0.0025 * h + 0.25 * (p - 10)
    
    # Test topography contour
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_topography_contour(x, y, h, ax=axes[0, 0], title='Topography Contours')
    plot_precipitation_contour(x, y, p, ax=axes[0, 1], title='Precipitation Contours')
    plot_wind_field(x, y, 10.0, 270.0, h, ax=axes[1, 0])
    
    # Test moisture ratio
    f_m = 1.0 - 0.5 * np.exp(-((X - 5000)**2) / 2e6)
    plot_moisture_ratio(x, y, f_m, ax=axes[1, 1], title='Moisture Ratio')
    
    plt.tight_layout()
    plt.savefig('test_map_plots.png', dpi=100)
    print("Test map plots saved to 'test_map_plots.png'")
    plt.close()
    
    # Test isotope maps
    fig = plot_isotope_maps(x, y, d2H, d18O, figsize=(12, 5))
    plt.savefig('test_isotope_maps.png', dpi=100)
    print("Test isotope maps saved to 'test_isotope_maps.png'")
    plt.close()
    
    print("\nAll map visualization tests completed!")
