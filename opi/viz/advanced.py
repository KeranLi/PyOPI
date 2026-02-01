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


def plot_cross_section(x, y, h_grid, d2h_grid=None, d18o_grid=None, 
                      section_y=None, figsize=(16, 12), save_path=None):
    """
    Plot cross-section through topography showing d2H and d18O isotopes.
    
    Parameters
    ----------
    x, y : ndarray
        Grid vectors (m)
    h_grid : ndarray
        Topography grid (m)
    d2h_grid : ndarray, optional
        d2H isotope grid (fraction, e.g., -0.0904 for -90.4 permil)
    d18o_grid : ndarray, optional
        d18O isotope grid (fraction, e.g., -0.0124 for -12.4 permil)
    section_y : float, optional
        Y-coordinate of cross-section. If None, use center.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, axes : matplotlib objects
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if section_y is None:
        section_y = 0
    
    # Find nearest y index
    y_idx = np.argmin(np.abs(y - section_y))
    
    # Extract cross-section
    h_section = h_grid[y_idx, :]
    
    # Create figure with gridspec for better layout control
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.2, 1.2], hspace=0.25)
    
    # ========== Subplot 1: 3D Topography ==========
    ax1 = fig.add_subplot(gs[0], projection='3d')
    X, Y = np.meshgrid(x/1000, y/1000)
    
    # Use terrain colormap for topography
    surf = ax1.plot_surface(X, Y, h_grid/1000, cmap='terrain', 
                            alpha=0.9, rstride=10, cstride=10, 
                            linewidth=0, antialiased=True)
    
    ax1.set_xlabel('x (km)', labelpad=5)
    ax1.set_ylabel('y (km)', labelpad=5)
    ax1.set_zlabel('Elevation (km)', labelpad=5)
    ax1.set_title('3D Topography')
    
    # Set consistent x limits with other subplots - full width
    ax1.set_xlim([np.min(x)/1000, np.max(x)/1000])
    ax1.set_ylim([np.min(y)/1000, np.max(y)/1000])
    ax1.set_zlim([0, 6])  # Max 6km to match other subplots
    
    # Adjust view angle to show long x direction (more elongated in x)
    ax1.view_init(elev=20, azim=-70)  # Moderate elevation for better view
    
    # Adjust the aspect ratio of the 3D plot box
    ax1.set_box_aspect([3, 1.5, 1])  # [x, y, z] aspect - wider in x
    
    # ========== Subplot 2: d2H Cross-section ==========
    ax2 = fig.add_subplot(gs[1])
    if d2h_grid is not None:
        d2h_section = d2h_grid[y_idx, :] * 1000  # Convert to permil
        # Auto-determine color range based on actual data
        vmin_d2h = np.floor(d2h_section.min() / 10) * 10  # Round down to nearest 10
        vmax_d2h = np.ceil(d2h_section.max() / 10) * 10   # Round up to nearest 10
        im2 = ax2.pcolormesh(x/1000, np.linspace(0, 6, 100), 
                            np.tile(d2h_section, (100, 1)), 
                            cmap='viridis_r', vmin=vmin_d2h, vmax=vmax_d2h)
        ax2.fill_between(x/1000, 0, h_section/1000, color='white', alpha=1.0)
        ax2.plot(x/1000, h_section/1000, 'k-', linewidth=1)
        ax2.set_ylabel('Height (km)')
        ax2.set_title(f'd2H Cross-section (permil) - Range: {d2h_section.min():.1f} to {d2h_section.max():.1f}')
        ax2.set_xlim([np.min(x)/1000, np.max(x)/1000])
        ax2.set_ylim([0, 6])  # Changed from 0-12 to 0-6 km
        plt.colorbar(im2, ax=ax2, label='d2H (permil)')
    else:
        ax2.text(0.5, 0.5, 'd2H data not available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_ylabel('Height (km)')
    
    # ========== Subplot 3: d18O Cross-section ==========
    ax3 = fig.add_subplot(gs[2])
    if d18o_grid is not None:
        d18o_section = d18o_grid[y_idx, :] * 1000  # Convert to permil
        # Auto-determine color range based on actual data
        vmin_d18o = np.floor(d18o_section.min() / 2) * 2  # Round down to nearest 2
        vmax_d18o = np.ceil(d18o_section.max() / 2) * 2   # Round up to nearest 2
        im3 = ax3.pcolormesh(x/1000, np.linspace(0, 6, 100), 
                            np.tile(d18o_section, (100, 1)), 
                            cmap='viridis_r', vmin=vmin_d18o, vmax=vmax_d18o)
        ax3.fill_between(x/1000, 0, h_section/1000, color='white', alpha=1.0)
        ax3.plot(x/1000, h_section/1000, 'k-', linewidth=1)
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('Height (km)')
        ax3.set_title(f'd18O Cross-section (permil) - Range: {d18o_section.min():.1f} to {d18o_section.max():.1f}')
        ax3.set_xlim([np.min(x)/1000, np.max(x)/1000])
        ax3.set_ylim([0, 6])  # Changed from 0-12 to 0-6 km
        plt.colorbar(im3, ax=ax3, label='d18O (permil)')
    else:
        ax3.text(0.5, 0.5, 'd18O data not available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=14)
        ax3.set_xlabel('Distance (km)')
        ax3.set_ylabel('Height (km)')
    
    # Note: tight_layout is skipped for 3D plots to avoid warnings
    # plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, [ax1, ax2, ax3]
