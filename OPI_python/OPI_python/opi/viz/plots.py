"""
Plotting functions for OPI results.

Standard plots for comparing samples, residuals, and meteoric water lines.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sample_comparison(sample_d2h, sample_d18o, pred_d2h, pred_d18o,
                          sample_type=None, title='Sample Comparison',
                          figsize=(12, 5), save_path=None):
    """
    Plot comparison between observed and predicted isotope values.
    
    Parameters
    ----------
    sample_d2h, sample_d18o : ndarray
        Observed isotope values (fraction)
    pred_d2h, pred_d18o : ndarray
        Predicted isotope values (fraction)
    sample_type : ndarray, optional
        Sample type labels
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
    
    # d2H comparison
    axes[0].scatter(sample_d2h * 1000, pred_d2h * 1000, c='blue', alpha=0.6)
    
    # 1:1 line
    min_val = min(np.min(sample_d2h), np.min(pred_d2h)) * 1000
    max_val = max(np.max(sample_d2h), np.max(pred_d2h)) * 1000
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
    
    axes[0].set_xlabel('Observed d2H (permil)')
    axes[0].set_ylabel('Predicted d2H (permil)')
    axes[0].set_title('d2H Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # d18O comparison
    axes[1].scatter(sample_d18o * 1000, pred_d18o * 1000, c='red', alpha=0.6)
    
    min_val = min(np.min(sample_d18o), np.min(pred_d18o)) * 1000
    max_val = max(np.max(sample_d18o), np.max(pred_d18o)) * 1000
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
    
    axes[1].set_xlabel('Observed d18O (permil)')
    axes[1].set_ylabel('Predicted d18O (permil)')
    axes[1].set_title('d18O Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_residuals(sample_d2h, sample_d18o, pred_d2h, pred_d18o,
                  elevation=None, title='Residuals Analysis',
                  figsize=(14, 5), save_path=None):
    """
    Plot residuals (observed - predicted) vs elevation or sample index.
    
    Parameters
    ----------
    sample_d2h, sample_d18o : ndarray
        Observed isotope values (fraction)
    pred_d2h, pred_d18o : ndarray
        Predicted isotope values (fraction)
    elevation : ndarray, optional
        Sample elevations (m)
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
    
    # Calculate residuals
    res_d2h = (sample_d2h - pred_d2h) * 1000  # Convert to permil
    res_d18o = (sample_d18o - pred_d18o) * 1000
    
    x_axis = elevation / 1000 if elevation is not None else np.arange(len(res_d2h))
    x_label = 'Elevation (km)' if elevation is not None else 'Sample Index'
    
    # d2H residuals
    axes[0].scatter(x_axis, res_d2h, c='blue', alpha=0.6)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('Residual d2H (permil)')
    axes[0].set_title('d2H Residuals')
    axes[0].grid(True, alpha=0.3)
    
    # d18O residuals
    axes[1].scatter(x_axis, res_d18o, c='red', alpha=0.6)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel('Residual d18O (permil)')
    axes[1].set_title('d18O Residuals')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, axes


def plot_mwl(d18o, d2h, title='Meteoric Water Line', 
            figsize=(8, 8), save_path=None):
    """
    Plot meteoric water line (d2H vs d18O).
    
    Parameters
    ----------
    d18o, d2h : ndarray
        Isotope values (fraction or permil)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to permil if needed
    if np.max(np.abs(d18o)) < 1:
        d18o = d18o * 1000
        d2h = d2h * 1000
    
    # Scatter plot
    ax.scatter(d18o, d2h, c='blue', alpha=0.6, s=50)
    
    # Global MWL (Craig, 1961)
    x_mwl = np.array([-25, 5])
    y_mwl = 8 * x_mwl + 10
    ax.plot(x_mwl, y_mwl, 'k--', label='GMWL (slope=8, intercept=10)')
    
    # Fit local MWL
    if len(d18o) > 2:
        coeffs = np.polyfit(d18o, d2h, 1)
        y_fit = coeffs[0] * x_mwl + coeffs[1]
        ax.plot(x_mwl, y_fit, 'r-', 
                label=f'Local MWL (slope={coeffs[0]:.2f}, intercept={coeffs[1]:.1f})')
    
    ax.set_xlabel('d18O (permil)')
    ax.set_ylabel('d2H (permil)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add d-excess contours
    x_range = np.linspace(np.min(d18o), np.max(d18o), 100)
    for d_excess in [-10, 0, 10, 20]:
        y_de = 8 * x_range + d_excess
        ax.plot(x_range, y_de, 'gray', alpha=0.3, linestyle=':')
        ax.text(x_range[-1], y_de[-1], f'd={d_excess}', 
                fontsize=8, alpha=0.5, ha='left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def plot_dexcess(d18o, d2h, elevation=None, title='d-Excess',
                figsize=(10, 5), save_path=None):
    """
    Plot d-excess values.
    
    Parameters
    ----------
    d18o, d2h : ndarray
        Isotope values (fraction or permil)
    elevation : ndarray, optional
        Sample elevations (m)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate d-excess
    if np.max(np.abs(d18o)) < 1:
        d_excess = (d2h * 1000) - 8 * (d18o * 1000)
    else:
        d_excess = d2h - 8 * d18o
    
    x_axis = elevation / 1000 if elevation is not None else np.arange(len(d_excess))
    x_label = 'Elevation (km)' if elevation is not None else 'Sample Index'
    
    ax.scatter(x_axis, d_excess, c='green', alpha=0.6, s=50)
    ax.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='Global average (10 permil)')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('d-excess (permil)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
