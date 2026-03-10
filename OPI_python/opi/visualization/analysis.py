"""
Analysis Visualization Functions for OPI

This module provides visualization functions for statistical analysis
of isotope data, including meteoric water line plots, residual analysis,
and observed vs predicted comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Tuple, Dict


def plot_meteoric_water_line(observed: Dict[str, np.ndarray],
                             predicted: Dict[str, np.ndarray],
                             ax: Optional[plt.Axes] = None,
                             title: Optional[str] = None,
                             show_stats: bool = True,
                             mwl_params: Optional[Dict[str, float]] = None) -> plt.Figure:
    """
    Create a Meteoric Water Line (MWL) plot comparing observed and predicted isotopes.
    
    Parameters
    ----------
    observed : dict
        Dictionary with 'd2H' and 'd18O' keys containing observed isotope values.
        Values can be in permil or fraction units.
    predicted : dict
        Dictionary with 'd2H' and 'd18O' keys containing predicted isotope values.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title. If None, uses default title.
    show_stats : bool, optional
        Whether to display MWL statistics on the plot. Default is True.
    mwl_params : dict, optional
        Pre-computed MWL parameters with 'slope' and 'intercept' keys.
        If None, MWL is calculated from observed data.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot
        
    Examples
    --------
    >>> observed = {'d2H': np.array([-70, -85, -55, -110, -75]),
    ...             'd18O': np.array([-10, -12, -8, -15, -11])}
    >>> predicted = {'d2H': np.array([-72, -83, -58, -105, -78]),
    ...              'd18O': np.array([-10.2, -11.8, -8.3, -14.5, -11.2])}
    >>> fig = plot_meteoric_water_line(observed, predicted)
    
    >>> # With custom title
    >>> fig = plot_meteoric_water_line(observed, predicted, 
    ...                                title='Sample Site MWL Analysis')
    """
    d2H_obs = np.asarray(observed['d2H'])
    d18O_obs = np.asarray(observed['d18O'])
    d2H_pred = np.asarray(predicted['d2H'])
    d18O_pred = np.asarray(predicted['d18O'])
    
    # Scale to permil if in fraction
    if np.max(np.abs(d2H_obs)) < 1:
        d2H_obs = d2H_obs * 1000
        d18O_obs = d18O_obs * 1000
        d2H_pred = d2H_pred * 1000
        d18O_pred = d18O_pred * 1000
    
    # Remove NaN values
    mask_obs = ~(np.isnan(d2H_obs) | np.isnan(d18O_obs))
    mask_pred = ~(np.isnan(d2H_pred) | np.isnan(d18O_pred))
    
    d2H_obs_clean = d2H_obs[mask_obs]
    d18O_obs_clean = d18O_obs[mask_obs]
    d2H_pred_clean = d2H_pred[mask_pred]
    d18O_pred_clean = d18O_pred[mask_pred]
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    # Calculate MWL from observed data if not provided
    if mwl_params is None:
        if len(d2H_obs_clean) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                d18O_obs_clean, d2H_obs_clean
            )
            mwl_params = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            }
        else:
            # Use global MWL as default
            mwl_params = {'slope': 8.0, 'intercept': 10.0}
    
    # Plot observed data
    ax.scatter(d18O_obs_clean, d2H_obs_clean, c='blue', s=80, alpha=0.6,
               edgecolors='darkblue', linewidths=1, label='Observed', zorder=3)
    
    # Plot predicted data
    ax.scatter(d18O_pred_clean, d2H_pred_clean, c='red', s=80, alpha=0.6,
               edgecolors='darkred', linewidths=1, marker='s', 
               label='Predicted', zorder=3)
    
    # Plot MWL
    x_range = np.array([min(d18O_obs_clean.min(), d18O_pred_clean.min()) - 2,
                        max(d18O_obs_clean.max(), d18O_pred_clean.max()) + 2])
    y_mwl = mwl_params['slope'] * x_range + mwl_params['intercept']
    ax.plot(x_range, y_mwl, 'k--', linewidth=2, 
            label=f"MWL: δ²H = {mwl_params['slope']:.2f}δ¹⁸O + {mwl_params['intercept']:.1f}‰",
            zorder=2)
    
    # Plot 1:1 line for reference
    min_val = min(x_range[0], (y_mwl[0] - mwl_params['intercept']) / mwl_params['slope'])
    max_val = max(x_range[1], (y_mwl[1] - mwl_params['intercept']) / mwl_params['slope'])
    ax.plot([min_val, max_val], [min_val * 8 + 10, max_val * 8 + 10], 
            'gray', linestyle=':', alpha=0.5, zorder=1)
    
    # Add statistics text
    if show_stats and 'r_value' in mwl_params:
        stats_text = (
            f"MWL Statistics:\n"
            f"Slope: {mwl_params['slope']:.2f}\n"
            f"Intercept: {mwl_params['intercept']:.1f}‰\n"
            f"R² = {mwl_params['r_value']**2:.3f}\n"
            f"n = {len(d2H_obs_clean)}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('δ¹⁸O (‰)', fontsize=12)
    ax.set_ylabel('δ²H (‰)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Meteoric Water Line Analysis', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    return fig


def plot_residuals(residuals: np.ndarray,
                   ax: Optional[plt.Axes] = None,
                   title: Optional[str] = None,
                   bins: int = 20,
                   show_stats: bool = True) -> plt.Figure:
    """
    Create a histogram of residuals with normal distribution overlay.
    
    Parameters
    ----------
    residuals : array_like
        Array of residual values (observed - predicted)
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title. If None, uses default title.
    bins : int, optional
        Number of bins for the histogram. Default is 20.
    show_stats : bool, optional
        Whether to display statistics on the plot. Default is True.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot
        
    Examples
    --------
    >>> residuals = np.random.normal(0, 1, 100)
    >>> fig = plot_residuals(residuals)
    
    >>> # Custom bins
    >>> fig = plot_residuals(residuals, bins=30, title='Model Residuals')
    """
    residuals = np.asarray(residuals)
    
    # Remove NaN values
    residuals_clean = residuals[~np.isnan(residuals)]
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    # Calculate statistics
    mean_res = np.mean(residuals_clean)
    std_res = np.std(residuals_clean)
    rmse = np.sqrt(np.mean(residuals_clean**2))
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(residuals_clean, bins=bins, 
                                         density=True, alpha=0.7, 
                                         color='steelblue', edgecolor='black')
    
    # Overlay normal distribution
    x_norm = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
    y_norm = stats.norm.pdf(x_norm, mean_res, std_res)
    ax.plot(x_norm, y_norm, 'r-', linewidth=2, 
            label=f'Normal (μ={mean_res:.2f}, σ={std_res:.2f})')
    
    # Add zero line
    ax.axvline(x=0, color='green', linestyle='--', linewidth=1.5, 
               label='Zero Reference')
    
    # Add mean line
    ax.axvline(x=mean_res, color='orange', linestyle='-.', linewidth=1.5,
               label=f'Mean ({mean_res:.2f})')
    
    # Add statistics text
    if show_stats:
        stats_text = (
            f"n = {len(residuals_clean)}\n"
            f"Mean = {mean_res:.3f}\n"
            f"Std = {std_res:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"Min = {residuals_clean.min():.3f}\n"
            f"Max = {residuals_clean.max():.3f}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel('Residual', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Residual Distribution', fontsize=14)
    
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def plot_observed_vs_predicted(observed: Dict[str, np.ndarray],
                               predicted: Dict[str, np.ndarray],
                               ax: Optional[plt.Axes] = None,
                               title: Optional[str] = None,
                               isotope: str = 'd2H',
                               show_stats: bool = True) -> plt.Figure:
    """
    Create a scatter plot of observed vs predicted values.
    
    Parameters
    ----------
    observed : dict
        Dictionary with isotope values (e.g., 'd2H' or 'd18O')
    predicted : dict
        Dictionary with predicted isotope values matching observed keys
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes or creates new figure.
    title : str, optional
        Plot title. If None, uses default title.
    isotope : str, optional
        Which isotope to plot ('d2H' or 'd18O'). Default is 'd2H'.
    show_stats : bool, optional
        Whether to display statistics on the plot. Default is True.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot
        
    Examples
    --------
    >>> observed = {'d2H': np.array([-70, -85, -55, -110, -75])}
    >>> predicted = {'d2H': np.array([-72, -83, -58, -105, -78])}
    >>> fig = plot_observed_vs_predicted(observed, predicted, isotope='d2H')
    
    >>> # For oxygen isotopes
    >>> observed = {'d18O': np.array([-10, -12, -8, -15, -11])}
    >>> predicted = {'d18O': np.array([-10.2, -11.8, -8.3, -14.5, -11.2])}
    >>> fig = plot_observed_vs_predicted(observed, predicted, isotope='d18O')
    """
    obs_vals = np.asarray(observed[isotope])
    pred_vals = np.asarray(predicted[isotope])
    
    # Scale to permil if in fraction
    if np.max(np.abs(obs_vals)) < 1:
        obs_vals = obs_vals * 1000
        pred_vals = pred_vals * 1000
    
    # Remove NaN values
    mask = ~(np.isnan(obs_vals) | np.isnan(pred_vals))
    obs_clean = obs_vals[mask]
    pred_clean = pred_vals[mask]
    
    # Create new figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure
    
    # Calculate statistics
    if len(obs_clean) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            obs_clean, pred_clean
        )
        rmse = np.sqrt(np.mean((obs_clean - pred_clean)**2))
        bias = np.mean(pred_clean - obs_clean)
    else:
        slope = intercept = r_value = rmse = bias = np.nan
    
    # Determine plot limits
    min_val = min(obs_clean.min(), pred_clean.min())
    max_val = max(obs_clean.max(), pred_clean.max())
    padding = (max_val - min_val) * 0.1
    plot_lim = [min_val - padding, max_val + padding]
    
    # Plot 1:1 line
    ax.plot(plot_lim, plot_lim, 'k--', linewidth=2, label='1:1 Line', zorder=1)
    
    # Plot data points
    ax.scatter(obs_clean, pred_clean, c='blue', s=80, alpha=0.6,
               edgecolors='darkblue', linewidths=1, zorder=2)
    
    # Plot regression line
    if len(obs_clean) > 1 and not np.isnan(slope):
        x_reg = np.array(plot_lim)
        y_reg = slope * x_reg + intercept
        ax.plot(x_reg, y_reg, 'r-', linewidth=2,
                label=f'Regression: y = {slope:.2f}x + {intercept:.1f}', zorder=3)
    
    # Add statistics text
    if show_stats and len(obs_clean) > 1:
        stats_text = (
            f"n = {len(obs_clean)}\n"
            f"R² = {r_value**2:.3f}\n"
            f"RMSE = {rmse:.2f}‰\n"
            f"Bias = {bias:.2f}‰\n"
            f"Slope = {slope:.3f}\n"
            f"Intercept = {intercept:.2f}‰"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set labels and title
    isotope_label = 'δ²H' if isotope == 'd2H' else 'δ¹⁸O'
    ax.set_xlabel(f'Observed {isotope_label} (‰)', fontsize=12)
    ax.set_ylabel(f'Predicted {isotope_label} (‰)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Observed vs Predicted {isotope_label}', fontsize=14)
    
    # Set equal limits and aspect ratio
    ax.set_xlim(plot_lim)
    ax.set_ylim(plot_lim)
    ax.set_aspect('equal', adjustable='box')
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_model_diagnostics(observed: Dict[str, np.ndarray],
                           predicted: Dict[str, np.ndarray],
                           figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
    """
    Create a comprehensive diagnostic plot with MWL, residuals, and obs vs pred.
    
    Parameters
    ----------
    observed : dict
        Dictionary with 'd2H' and 'd18O' keys containing observed isotope values
    predicted : dict
        Dictionary with 'd2H' and 'd18O' keys containing predicted isotope values
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (14, 10).
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing all diagnostic plots
        
    Examples
    --------
    >>> observed = {'d2H': np.array([-70, -85, -55, -110, -75]),
    ...             'd18O': np.array([-10, -12, -8, -15, -11])}
    >>> predicted = {'d2H': np.array([-72, -83, -58, -105, -78]),
    ...              'd18O': np.array([-10.2, -11.8, -8.3, -14.5, -11.2])}
    >>> fig = plot_model_diagnostics(observed, predicted)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # MWL plot
    plot_meteoric_water_line(observed, predicted, ax=axes[0, 0],
                             title='Meteoric Water Line')
    
    # Observed vs Predicted for d2H
    plot_observed_vs_predicted(observed, predicted, ax=axes[0, 1],
                               isotope='d2H', title='Observed vs Predicted (δ²H)')
    
    # Observed vs Predicted for d18O
    plot_observed_vs_predicted(observed, predicted, ax=axes[1, 0],
                               isotope='d18O', title='Observed vs Predicted (δ¹⁸O)')
    
    # Calculate and plot residuals (in d2H space)
    d2H_obs = np.asarray(observed['d2H'])
    d2H_pred = np.asarray(predicted['d2H'])
    if np.max(np.abs(d2H_obs)) < 1:
        d2H_obs = d2H_obs * 1000
        d2H_pred = d2H_pred * 1000
    
    mask = ~(np.isnan(d2H_obs) | np.isnan(d2H_pred))
    residuals = d2H_obs[mask] - d2H_pred[mask]
    
    plot_residuals(residuals, ax=axes[1, 1], title='Residual Distribution (δ²H)')
    
    plt.tight_layout()
    fig.suptitle('Model Diagnostic Plots', fontsize=16, fontweight='bold', y=1.02)
    
    return fig


if __name__ == "__main__":
    # Test analysis visualization functions
    print("Testing analysis visualization functions...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    
    # Observed isotopes with some noise
    d18O_obs = np.random.uniform(-15, -8, n_samples)
    # GMWL: d2H = 8*d18O + 10 with noise
    d2H_obs = 8.0 * d18O_obs + 10.0 + np.random.normal(0, 3, n_samples)
    
    # Predicted isotopes (with some model bias)
    d18O_pred = d18O_obs + np.random.normal(0, 0.3, n_samples)
    d2H_pred = d2H_obs + np.random.normal(0.5, 2, n_samples)
    
    observed = {'d2H': d2H_obs, 'd18O': d18O_obs}
    predicted = {'d2H': d2H_pred, 'd18O': d18O_pred}
    
    # Test plot_meteoric_water_line
    fig = plot_meteoric_water_line(observed, predicted,
                                   title='Test Meteoric Water Line')
    plt.savefig('test_mwl.png', dpi=100)
    print("MWL plot saved to 'test_mwl.png'")
    plt.close()
    
    # Test plot_observed_vs_predicted
    fig = plot_observed_vs_predicted(observed, predicted, isotope='d2H',
                                     title='Test Observed vs Predicted')
    plt.savefig('test_obs_vs_pred.png', dpi=100)
    print("Observed vs Predicted plot saved to 'test_obs_vs_pred.png'")
    plt.close()
    
    # Test plot_residuals
    d2H_obs_arr = np.asarray(observed['d2H'])
    d2H_pred_arr = np.asarray(predicted['d2H'])
    residuals = d2H_obs_arr - d2H_pred_arr
    fig = plot_residuals(residuals, title='Test Residuals')
    plt.savefig('test_residuals.png', dpi=100)
    print("Residuals plot saved to 'test_residuals.png'")
    plt.close()
    
    # Test plot_model_diagnostics
    fig = plot_model_diagnostics(observed, predicted)
    plt.savefig('test_diagnostics.png', dpi=100)
    print("Diagnostic plots saved to 'test_diagnostics.png'")
    plt.close()
    
    print("\nAll analysis visualization tests completed!")
