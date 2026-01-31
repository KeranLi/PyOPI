"""
Advanced visualization functions for OPI.

Pair plots and prediction visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_pair_plots(data_dict, variables=None, figsize=(12, 12), save_path=None):
    """
    Create pair plots (scatter matrix) showing relationships between variables.
    
    Similar to MATLAB's opiPairPlots.m
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with variable names as keys and data arrays as values
    variables : list, optional
        List of variables to include. If None, use all.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib objects
    """
    if variables is None:
        variables = list(data_dict.keys())
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=figsize)
    
    # Ensure axes is 2D even for single variable
    if n_vars == 1:
        axes = np.array([[axes]])
    
    for i, var_row in enumerate(variables):
        for j, var_col in enumerate(variables):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                data = data_dict[var_row]
                ax.hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                ax.set_title(var_row)
            else:
                # Off-diagonal: scatter plot
                x_data = data_dict[var_col]
                y_data = data_dict[var_row]
                ax.scatter(x_data, y_data, c='blue', alpha=0.5, s=20)
                
                # Add trend line
                if len(x_data) > 1:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                    ax.plot(x_line, p(x_line), 'r--', alpha=0.8, linewidth=1)
                    
                    # Calculate R²
                    y_pred = p(x_data)
                    ss_res = np.sum((y_data - y_pred) ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    ax.text(0.05, 0.95, f'R²={r_squared:.3f}', 
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.5))
            
            # Hide x labels for non-bottom row
            if i < n_vars - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(var_col)
            
            # Hide y labels for non-left column
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(var_row)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_prediction(x_data, y_observed, y_predicted, x_label='X',
                   y_label='Y', title='Prediction vs Observation',
                   figsize=(12, 5), save_path=None):
    """
    Plot predictions against observations.
    
    Similar to MATLAB's opiPredictPlot.m
    
    Parameters
    ----------
    x_data : ndarray
        X-axis data (e.g., distance or elevation)
    y_observed : ndarray
        Observed values
    y_predicted : ndarray
        Predicted values
    x_label, y_label : str
        Axis labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Time series / profile plot
    ax = axes[0]
    ax.plot(x_data, y_observed, 'bo-', label='Observed', alpha=0.7)
    ax.plot(x_data, y_predicted, 'rs-', label='Predicted', alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{title} - Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot with 1:1 line
    ax = axes[1]
    ax.scatter(y_observed, y_predicted, c='blue', alpha=0.6, s=50)
    
    min_val = min(np.min(y_observed), np.min(y_predicted))
    max_val = max(np.max(y_observed), np.max(y_predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
    
    # Calculate and display statistics
    residuals = y_observed - y_predicted
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    
    ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nBias: {bias:.3f}',
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round',
           facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(f'Observed {y_label}')
    ax.set_ylabel(f'Predicted {y_label}')
    ax.set_title(f'{title} - Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_cross_section(x, y, h_grid, variable_grid, section_y=None,
                      variable_name='Variable', cmap='RdYlBu_r',
                      figsize=(12, 6), save_path=None):
    """
    Plot cross-section through topography showing a variable.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors (m)
    h_grid : ndarray
        Topography grid (m)
    variable_grid : ndarray
        Variable to plot (e.g., precipitation, d2H)
    section_y : float, optional
        Y-coordinate of cross-section. If None, use center.
    variable_name : str
        Name of variable for labels
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib objects
    """
    if section_y is None:
        section_y = 0
    
    # Find nearest y index
    y_idx = np.argmin(np.abs(y - section_y))
    
    # Extract cross-sections
    h_section = h_grid[y_idx, :]
    var_section = variable_grid[y_idx, :]
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, 
                            gridspec_kw={'height_ratios': [1, 2]})
    
    # Topography profile
    ax = axes[0]
    ax.fill_between(x/1000, 0, h_section/1000, color='brown', alpha=0.7)
    ax.set_ylabel('Elevation (km)')
    ax.set_title(f'Topography at y={section_y/1000:.1f} km')
    ax.set_xlim([np.min(x)/1000, np.max(x)/1000])
    ax.grid(True, alpha=0.3)
    
    # Variable profile
    ax = axes[1]
    im = ax.pcolormesh(x/1000, np.linspace(0, 12, 100), 
                       np.tile(var_section, (100, 1)), cmap=cmap)
    ax.fill_between(x/1000, 0, h_section/1000, color='white', alpha=1.0)
    ax.plot(x/1000, h_section/1000, 'k-', linewidth=1)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Height (km)')
    ax.set_title(f'{variable_name} Cross-section')
    ax.set_xlim([np.min(x)/1000, np.max(x)/1000])
    ax.set_ylim([0, 12])
    plt.colorbar(im, ax=ax, label=variable_name)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes
