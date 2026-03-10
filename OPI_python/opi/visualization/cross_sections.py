"""
Cross-Section Visualization Functions for OPI

This module provides functions for extracting and visualizing cross-sections
through 2D grids of topography, precipitation, and isotope data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from typing import Optional, Tuple, Union, List


def extract_cross_section(grid: np.ndarray,
                          x: np.ndarray,
                          y: np.ndarray,
                          start_point: Tuple[float, float],
                          end_point: Tuple[float, float],
                          n_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract values along a cross-section line from a 2D grid.
    
    Parameters
    ----------
    grid : array_like
        2D array of values with shape (len(y), len(x))
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    start_point : tuple
        Starting point of cross-section as (x, y) coordinates
    end_point : tuple
        Ending point of cross-section as (x, y) coordinates
    n_points : int, optional
        Number of points along cross-section. If None, uses max(len(x), len(y)).
        
    Returns
    -------
    tuple
        (distances, coordinates, values) where:
        - distances: 1D array of distances along cross-section (m)
        - coordinates: 2D array of (x, y) coordinates along cross-section
        - values: 1D array of interpolated grid values
        
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    >>> dist, coords, h_cs = extract_cross_section(h, x, y, (0, 5000), (10000, 5000))
    >>> print(f"Cross-section length: {dist[-1]:.0f} m")
    """
    grid = np.asarray(grid)
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Determine number of points
    if n_points is None:
        n_points = max(len(x), len(y))
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (y, x), grid, 
        method='linear', 
        bounds_error=False, 
        fill_value=np.nan
    )
    
    # Generate points along cross-section line
    x_start, y_start = start_point
    x_end, y_end = end_point
    
    x_points = np.linspace(x_start, x_end, n_points)
    y_points = np.linspace(y_start, y_end, n_points)
    coordinates = np.column_stack([x_points, y_points])
    
    # Calculate distances along cross-section
    dx = x_points - x_start
    dy = y_points - y_start
    distances = np.sqrt(dx**2 + dy**2)
    
    # Interpolate values (note: interpolator expects (y, x) order)
    points_to_interp = np.column_stack([y_points, x_points])
    values = interpolator(points_to_interp)
    
    return distances, coordinates, values


def plot_cross_section(x: np.ndarray,
                       y: np.ndarray,
                       h_grid: np.ndarray,
                       p_grid: np.ndarray,
                       iso_grid: np.ndarray,
                       start_point: Tuple[float, float],
                       end_point: Tuple[float, float],
                       ax: Optional[plt.Axes] = None,
                       title: Optional[str] = None,
                       show_topography: bool = True,
                       n_points: Optional[int] = None) -> plt.Figure:
    """
    Create a cross-section plot showing topography, precipitation, and isotopes.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    h_grid : array_like
        2D array of elevation values (m)
    p_grid : array_like
        2D array of precipitation values
    iso_grid : array_like
        2D array of isotope values (d2H or d18O)
    start_point : tuple
        Starting point of cross-section as (x, y) coordinates
    end_point : tuple
        Ending point of cross-section as (x, y) coordinates
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title
    show_topography : bool, optional
        Whether to show topography as filled area below the cross-section
    n_points : int, optional
        Number of points along cross-section. If None, uses max(len(x), len(y)).
        
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
    >>> p = 10 * np.exp(-((X - 5000)**2) / 2e6) * (1 + 0.5 * h / 1000)
    >>> iso = -100 + 0.02 * h
    >>> fig = plot_cross_section(x, y, h, p, iso, (0, 5000), (10000, 5000))
    
    >>> # Custom title without topography fill
    >>> fig = plot_cross_section(x, y, h, p, iso, (0, 5000), (10000, 5000),
    ...                          title='West-East Cross-Section', 
    ...                          show_topography=False)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    h_grid = np.asarray(h_grid)
    p_grid = np.asarray(p_grid)
    iso_grid = np.asarray(iso_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Extract cross-sections
    dist_h, _, h_cs = extract_cross_section(h_grid, x, y, start_point, end_point, n_points)
    dist_p, _, p_cs = extract_cross_section(p_grid, x, y, start_point, end_point, n_points)
    dist_iso, _, iso_cs = extract_cross_section(iso_grid, x, y, start_point, end_point, n_points)
    
    # Use distances from topography (should be the same)
    distances = dist_h
    
    # Create secondary axes for precipitation and isotopes
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    
    # Plot topography
    if show_topography:
        ax.fill_between(distances, 0, h_cs, alpha=0.3, color='brown', label='Topography')
    ax.plot(distances, h_cs, 'brown', linewidth=2, label='Elevation')
    
    # Plot precipitation
    line2 = ax2.plot(distances, p_cs, 'b-', linewidth=2, label='Precipitation')
    ax2.set_ylabel('Precipitation (mm/day)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Plot isotopes
    line3 = ax3.plot(distances, iso_cs * 1000, 'r-', linewidth=2, label='Isotopes')
    ax3.set_ylabel('Isotope Value (‰)', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    
    # Set labels and title
    ax.set_xlabel('Distance Along Cross-Section (m)')
    ax.set_ylabel('Elevation (m)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Cross-Section: ({start_point[0]:.0f}, {start_point[1]:.0f}) to '
                     f'({end_point[0]:.0f}, {end_point[1]:.0f})')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + line2 + line3, labels1 + ['Precipitation', 'Isotopes'],
              loc='upper right')
    
    return fig


def plot_multiple_cross_sections(x: np.ndarray,
                                  y: np.ndarray,
                                  grid_dict: dict,
                                  start_point: Tuple[float, float],
                                  end_point: Tuple[float, float],
                                  figsize: Tuple[float, float] = (14, 10),
                                  n_points: Optional[int] = None) -> plt.Figure:
    """
    Create multiple cross-section subplots for different variables.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    grid_dict : dict
        Dictionary mapping variable names to 2D arrays
    start_point : tuple
        Starting point of cross-section as (x, y) coordinates
    end_point : tuple
        Ending point of cross-section as (x, y) coordinates
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (14, 10).
    n_points : int, optional
        Number of points along cross-section.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing all subplots
        
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    >>> p = 10 * np.exp(-((X - 5000)**2) / 2e6)
    >>> grids = {'Topography': h, 'Precipitation': p}
    >>> fig = plot_multiple_cross_sections(x, y, grids, (0, 5000), (10000, 5000))
    """
    n_plots = len(grid_dict)
    
    if n_plots == 0:
        raise ValueError("grid_dict must contain at least one grid")
    
    # Calculate figure layout
    n_cols = min(n_plots, 2)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Extract topography if present for background
    if 'Topography' in grid_dict or 'h_grid' in grid_dict:
        h_key = 'Topography' if 'Topography' in grid_dict else 'h_grid'
        h_grid = grid_dict[h_key]
        dist_h, _, h_cs = extract_cross_section(h_grid, x, y, start_point, end_point, n_points)
    else:
        dist_h = None
    
    # Plot each variable
    for idx, (name, grid) in enumerate(grid_dict.items()):
        ax = axes[idx]
        
        distances, _, values = extract_cross_section(grid, x, y, start_point, end_point, n_points)
        
        # If this is topography, plot differently
        if name in ['Topography', 'h_grid', 'Elevation']:
            ax.fill_between(distances, 0, values, alpha=0.3, color='brown')
            ax.plot(distances, values, 'brown', linewidth=2)
            ax.set_ylabel('Elevation (m)')
        else:
            # Plot regular variable
            ax.plot(distances, values, linewidth=2)
            ax.set_ylabel(name)
            
            # Add topography background if available
            if dist_h is not None and name not in ['Topography', 'h_grid', 'Elevation']:
                ax2 = ax.twinx()
                ax2.fill_between(dist_h, 0, h_cs, alpha=0.15, color='brown')
                ax2.set_ylabel('Elevation (m)', color='brown')
                ax2.tick_params(axis='y', labelcolor='brown')
        
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    # Set common x-label
    for ax in axes[-n_cols:]:
        ax.set_xlabel('Distance Along Cross-Section (m)')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle(f'Cross-Sections: ({start_point[0]:.0f}, {start_point[1]:.0f}) to '
                 f'({end_point[0]:.0f}, {end_point[1]:.0f})',
                 fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Test cross-section visualization functions
    print("Testing cross-section visualization functions...")
    
    # Create sample data
    x = np.linspace(0, 10000, 100)
    y = np.linspace(0, 10000, 100)
    X, Y = np.meshgrid(x, y)
    
    # Topography: Gaussian mountain
    h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    
    # Precipitation: orographic enhancement
    p = 10 * np.exp(-((X - 5000)**2) / 2e6) * (1 + 0.5 * h / 1000)
    
    # Isotopes: altitude effect
    iso = -100 + 0.02 * h
    
    # Test extract_cross_section
    dist, coords, h_cs = extract_cross_section(h, x, y, (0, 5000), (10000, 5000))
    print(f"Cross-section length: {dist[-1]:.0f} m")
    print(f"Number of points: {len(dist)}")
    print(f"Elevation range: {h_cs.min():.1f} to {h_cs.max():.1f} m")
    
    # Test plot_cross_section
    fig = plot_cross_section(x, y, h, p, iso, (0, 5000), (10000, 5000),
                            title='West-East Cross-Section')
    plt.savefig('test_cross_section.png', dpi=100)
    print("Cross-section plot saved to 'test_cross_section.png'")
    plt.close()
    
    # Test plot_multiple_cross_sections
    grids = {
        'Topography': h,
        'Precipitation': p,
        'd2H': iso
    }
    fig = plot_multiple_cross_sections(x, y, grids, (2500, 2500), (7500, 7500))
    plt.savefig('test_multiple_cross_sections.png', dpi=100)
    print("Multiple cross-sections plot saved to 'test_multiple_cross_sections.png'")
    plt.close()
    
    print("\nAll cross-section visualization tests completed!")
