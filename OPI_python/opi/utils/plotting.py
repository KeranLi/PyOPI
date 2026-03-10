"""
Plotting utility functions for OPI

This module provides basic plotting functions for visualizing
orographic precipitation and isotope model results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_topography(x, y, h_grid, ax=None, title=None, cmap='terrain',
                    colorbar=True, units='m', **kwargs):
    """
    Plot topography as a 2D color map.
    
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
    title : str, optional
        Plot title
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for topography (default: 'terrain')
    colorbar : bool, optional
        Whether to add a colorbar (default: True)
    units : str, optional
        Units for elevation values in colorbar label (default: 'm')
    **kwargs : dict
        Additional keyword arguments passed to imshow()
    
    Returns
    -------
    matplotlib.image.AxesImage
        The image object returned by imshow
    
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> h = 1000 * np.exp(-((x[:, None] - 5000)**2 + (y[None, :] - 5000)**2) / 1e6)
    >>> plot_topography(x, y, h, title='Mountain Topography')
    
    >>> # Plot on existing axes
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> plot_topography(x, y, h, ax=axes[0], title='Topography')
    """
    x = np.asarray(x)
    y = np.asarray(y)
    h_grid = np.asarray(h_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set default imshow arguments
    imshow_kwargs = {
        'origin': 'lower',
        'extent': [x.min(), x.max(), y.min(), y.max()],
        'cmap': cmap,
        'aspect': 'auto'
    }
    imshow_kwargs.update(kwargs)
    
    # Create the plot
    im = ax.imshow(h_grid, **imshow_kwargs)
    
    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'Elevation ({units})')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if title:
        ax.set_title(title)
    
    return im


def plot_precipitation(x, y, p_grid, ax=None, title=None, cmap='Blues',
                       colorbar=True, units='mm/day', **kwargs):
    """
    Plot precipitation field as a 2D color map.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    p_grid : array_like
        2D array of precipitation values with shape (len(y), len(x))
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for precipitation (default: 'Blues')
    colorbar : bool, optional
        Whether to add a colorbar (default: True)
    units : str, optional
        Units for precipitation values in colorbar label (default: 'mm/day')
    **kwargs : dict
        Additional keyword arguments passed to imshow()
    
    Returns
    -------
    matplotlib.image.AxesImage
        The image object returned by imshow
    
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> p = 10 * np.exp(-((x[:, None] - 5000)**2) / 1e6)
    >>> plot_precipitation(x, y, p, title='Orographic Precipitation')
    
    >>> # Plot on existing axes
    >>> fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    >>> plot_precipitation(x, y, p, ax=axes[0], title='Precipitation Field')
    """
    x = np.asarray(x)
    y = np.asarray(y)
    p_grid = np.asarray(p_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set default imshow arguments
    imshow_kwargs = {
        'origin': 'lower',
        'extent': [x.min(), x.max(), y.min(), y.max()],
        'cmap': cmap,
        'aspect': 'auto'
    }
    imshow_kwargs.update(kwargs)
    
    # Create the plot
    im = ax.imshow(p_grid, **imshow_kwargs)
    
    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'Precipitation ({units})')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if title:
        ax.set_title(title)
    
    return im


def plot_isotopes(x, y, iso_grid, ax=None, title=None, cmap='RdBu_r',
                  colorbar=True, units='‰', isotope='δ²H', **kwargs):
    """
    Plot isotope field as a 2D color map.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    iso_grid : array_like
        2D array of isotope values with shape (len(y), len(x))
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title. If None, a default title is generated based on the isotope type.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for isotope values (default: 'RdBu_r' - diverging red-blue)
    colorbar : bool, optional
        Whether to add a colorbar (default: True)
    units : str, optional
        Units for isotope values in colorbar label (default: '‰')
    isotope : str, optional
        Isotope type label for title/colorbar (default: 'δ²H')
    **kwargs : dict
        Additional keyword arguments passed to imshow()
    
    Returns
    -------
    matplotlib.image.AxesImage
        The image object returned by imshow
    
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> iso = -100 + 20 * np.sin(x[:, None] / 1000) * np.cos(y[None, :] / 1000)
    >>> plot_isotopes(x, y, iso, title='Hydrogen Isotope Distribution')
    
    >>> # Plot oxygen isotopes
    >>> iso_18o = -12 + 2 * np.sin(x[:, None] / 1000)
    >>> plot_isotopes(x, y, iso_18o, isotope='δ¹⁸O', title='Oxygen Isotope Distribution')
    """
    x = np.asarray(x)
    y = np.asarray(y)
    iso_grid = np.asarray(iso_grid)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine symmetric color limits for diverging colormap
    if 'vmin' not in kwargs and 'vmax' not in kwargs:
        vmax = np.max(np.abs([iso_grid.min(), iso_grid.max()]))
        if not np.isnan(vmax) and vmax > 0:
            imshow_kwargs = {'vmin': -vmax, 'vmax': vmax}
        else:
            imshow_kwargs = {}
    else:
        imshow_kwargs = {}
    
    # Set default imshow arguments
    imshow_kwargs.update({
        'origin': 'lower',
        'extent': [x.min(), x.max(), y.min(), y.max()],
        'cmap': cmap,
        'aspect': 'auto'
    })
    imshow_kwargs.update(kwargs)
    
    # Create the plot
    im = ax.imshow(iso_grid, **imshow_kwargs)
    
    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{isotope} ({units})')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{isotope} Isotope Distribution')
    
    return im


def plot_cross_section(x, grid_values, y_coord=None, ax=None, title=None,
                       xlabel='X (m)', ylabel='Value', label=None, **kwargs):
    """
    Plot a cross-section through a 2D grid.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    grid_values : array_like
        1D array of values along the cross-section
    y_coord : float, optional
        Y-coordinate of the cross-section for plot label
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis (default: 'X (m)')
    ylabel : str, optional
        Label for y-axis (default: 'Value')
    label : str, optional
        Line label for legend
    **kwargs : dict
        Additional keyword arguments passed to plot()
    
    Returns
    -------
    matplotlib.lines.Line2D
        The line object returned by plot
    
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> h = 1000 * np.exp(-(x - 5000)**2 / 1e6)
    >>> plot_cross_section(x, h, y_coord=5000, title='Topography Cross-section')
    """
    x = np.asarray(x)
    grid_values = np.asarray(grid_values)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set default plot arguments
    plot_kwargs = {'linewidth': 2}
    if label is None and y_coord is not None:
        plot_kwargs['label'] = f'y = {y_coord:.0f} m'
    elif label is not None:
        plot_kwargs['label'] = label
    plot_kwargs.update(kwargs)
    
    # Create the plot
    line = ax.plot(x, grid_values, **plot_kwargs)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    if 'label' in plot_kwargs:
        ax.legend()
    
    return line[0] if line else None


def plot_comparison(x, y, grids_dict, titles=None, cmap='viridis',
                    main_title=None, figsize=None):
    """
    Plot multiple 2D grids side by side for comparison.
    
    Parameters
    ----------
    x : array_like
        1D array of x-coordinates (m)
    y : array_like
        1D array of y-coordinates (m)
    grids_dict : dict
        Dictionary mapping grid names to 2D arrays
    titles : dict, optional
        Dictionary mapping grid names to plot titles
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for all plots (default: 'viridis')
    main_title : str, optional
        Overall figure title
    figsize : tuple, optional
        Figure size (width, height). If None, calculated based on number of plots.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plots
    
    Examples
    --------
    >>> x = np.linspace(0, 10000, 100)
    >>> y = np.linspace(0, 10000, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    >>> p = 10 * np.exp(-((X - 5000)**2) / 1e6)
    >>> grids = {'Topography': h, 'Precipitation': p}
    >>> plot_comparison(x, y, grids, titles={'Topography': 'Elevation (m)'})
    """
    n_plots = len(grids_dict)
    
    if n_plots == 0:
        raise ValueError("grids_dict must contain at least one grid")
    
    # Calculate figure layout
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    extent = [x.min(), x.max(), y.min(), y.max()]
    
    for idx, (name, grid) in enumerate(grids_dict.items()):
        ax = axes[idx]
        grid = np.asarray(grid)
        
        im = ax.imshow(grid, origin='lower', extent=extent, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax)
        
        # Set title
        if titles and name in titles:
            ax.set_title(titles[name])
        else:
            ax.set_title(name)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    if main_title:
        fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig


def plot_mwl(d18O, d2H, slope=None, intercept=None, ax=None, title=None,
             show_stats=True, **kwargs):
    """
    Plot isotope data on a delta^18O vs delta^2H scatter plot with MWL.
    
    Parameters
    ----------
    d18O : array_like
        delta^18O values (in permil)
    d2H : array_like
        delta^2H values (in permil)
    slope : float, optional
        MWL slope. If None, no line is drawn.
    intercept : float, optional
        MWL intercept. Required if slope is provided.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title
    show_stats : bool, optional
        Whether to display MWL statistics on the plot (default: True)
    **kwargs : dict
        Additional keyword arguments passed to scatter()
    
    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter plot object
    
    Examples
    --------
    >>> d18O = np.array([-10, -12, -8, -15, -11])
    >>> d2H = np.array([-70, -85, -55, -110, -75])
    >>> plot_mwl(d18O, d2H, slope=8.0, intercept=10.0, title='Meteoric Water Line')
    """
    d18O = np.asarray(d18O)
    d2H = np.asarray(d2H)
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set default scatter arguments
    scatter_kwargs = {'alpha': 0.6, 'edgecolors': 'black', 'linewidths': 0.5}
    scatter_kwargs.update(kwargs)
    
    # Create scatter plot
    scatter = ax.scatter(d18O, d2H, **scatter_kwargs)
    
    # Add MWL if provided
    if slope is not None and intercept is not None:
        x_line = np.array([d18O.min(), d18O.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'MWL: δ²H = {slope:.2f}δ¹⁸O + {intercept:.1f}‰')
        ax.legend()
        
        # Add statistics text
        if show_stats:
            from .statistics import estimate_mwl
            stats = estimate_mwl(d18O, d2H)
            stats_text = f"R² = {stats['r_value']**2:.3f}\np = {stats['p_value']:.4f}"
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set labels and title
    ax.set_xlabel('δ¹⁸O (‰)')
    ax.set_ylabel('δ²H (‰)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Isotope Data with Meteoric Water Line')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for proper visualization
    ax.set_aspect('equal', adjustable='box')
    
    return scatter


if __name__ == "__main__":
    # Test plotting functions
    print("Testing plotting utility functions...")
    
    # Create sample data
    x = np.linspace(0, 10000, 100)
    y = np.linspace(0, 10000, 100)
    X, Y = np.meshgrid(x, y)
    
    # Topography: Gaussian mountain
    h = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / 1e6)
    
    # Precipitation: orographic enhancement
    p = 10 * np.exp(-((X - 5000)**2) / 2e6) * (1 + 0.5 * h / 1000)
    
    # Isotopes: altitude and amount effect
    iso = -100 + 0.02 * h + 2 * (p - 10)
    
    # Test individual plot functions
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_topography(x, y, h, ax=axes[0, 0], title='Topography')
    plot_precipitation(x, y, p, ax=axes[0, 1], title='Precipitation')
    plot_isotopes(x, y, iso, ax=axes[1, 0], title='Isotope Distribution')
    
    # Test cross-section
    center_idx = len(y) // 2
    plot_cross_section(x, h[center_idx, :], y_coord=y[center_idx],
                       ax=axes[1, 1], title='Cross-section', ylabel='Elevation (m)')
    
    plt.tight_layout()
    plt.savefig('test_plots.png', dpi=100)
    print("Test plots saved to 'test_plots.png'")
    
    plt.close('all')
    
    print("\nAll plotting tests completed!")
