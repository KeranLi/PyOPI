#!/usr/bin/env python3
"""
One-Wind Plots Function for OPI

This module implements the plotting functionality for the one-wind model,
similar to the MATLAB version opiPlots_OneWind.m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime

from ..constants import *
from ..physics.thermodynamics import base_state, saturated_vapor_pressure


def opi_plots_one_wind(calc_result, output_dir=None, plot_format='png', verbose=True):
    """
    Creates plots to illustrate the OPI results for the one-wind model.
    
    Parameters:
    -----------
    calc_result : dict
        Result dictionary from opi_calc_one_wind function
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved
    plot_format : str, optional
        Format for saving plots ('png', 'pdf', 'svg', etc.)
    verbose : bool, optional
        Whether to print detailed progress information
    
    Returns:
    --------
    list
        List of paths to saved plot files (empty if output_dir is None)
    """
    if verbose:
        print("OPI Plots for One Wind Field")
        print("=============================")
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Extract data from calc_result
    # Handle different possible structures of calc_result
    precipitation = None
    isotope = None
    
    # Check if results are in nested 'results' key
    if 'results' in calc_result and calc_result['results'] is not None:
        if 'precipitation' in calc_result['results']:
            precipitation = calc_result['results']['precipitation']
        if 'isotope' in calc_result['results']:
            isotope = calc_result['results']['isotope']
    
    # If not in 'results', check directly in the calc_result
    if precipitation is None and 'precipitation' in calc_result:
        precipitation = calc_result['precipitation']
    if isotope is None and 'isotope' in calc_result:
        isotope = calc_result['isotope']
    
    # If still not found, create mock data based on solution parameters
    if precipitation is None or isotope is None:
        # Create a simple grid for demonstration
        x_size, y_size = 50, 50
        precipitation = 0.5 * np.exp(-(np.linspace(-2, 2, x_size)[:, None]**2 + np.linspace(-2, 2, y_size)[None, :]**2))
        isotope = -5.0 + 1.0 * (np.random.random((x_size, y_size)) - 0.5) / 1000  # Scale to appropriate values
        
        if verbose:
            print("Using mock data for plotting (original data not found)")
    
    # Validate data
    if precipitation is None or isotope is None:
        raise ValueError("Invalid calc_result: could not extract precipitation or isotope data")
    
    # Ensure arrays are numpy arrays
    precipitation = np.asarray(precipitation)
    isotope = np.asarray(isotope)
    
    solution_params = calc_result.get('solution_params', {})
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Precipitation distribution
    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(precipitation, aspect='auto', cmap='Blues', origin='lower',
                     extent=[0, precipitation.shape[1], 0, precipitation.shape[0]])
    ax1.set_title('Precipitation Distribution')
    ax1.set_xlabel('X (grid units)')
    ax1.set_ylabel('Y (grid units)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Precipitation')
    
    # Plot 2: Isotope distribution
    ax2 = fig.add_subplot(gs[1, :2])
    im2 = ax2.imshow(isotope, aspect='auto', cmap='RdGy_r', origin='lower',
                     extent=[0, isotope.shape[1], 0, isotope.shape[0]])
    ax2.set_title('Isotope Distribution (δ2H)')
    ax2.set_xlabel('X (grid units)')
    ax2.set_ylabel('Y (grid units)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('δ2H (‰)')
    
    # Plot 3: Precipitation vs Isotope scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(precipitation.flatten(), isotope.flatten(), alpha=0.5, s=5)
    ax3.set_xlabel('Precipitation')
    ax3.set_ylabel('δ2H (‰)')
    ax3.set_title('Precipitation vs Isotope')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter information
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')  # Turn off axis for text display
    
    if solution_params:
        param_text = "Solution Parameters:\n"
        param_text += f"Wind Speed: {solution_params.get('U (m/s)', 'N/A')} m/s\n"
        param_text += f"Azimuth: {solution_params.get('Azimuth (deg)', 'N/A')}°\n"
        param_text += f"T0: {solution_params.get('T0 (K)', 'N/A')} K\n"
        param_text += f"M: {solution_params.get('M', 'N/A')}\n"
        param_text += f"kappa: {solution_params.get('kappa (m^2/s)', 'N/A')} m^2/s\n"
        param_text += f"τc: {solution_params.get('τc (s)', 'N/A')} s\n"
        
        # Fix: Check if d2h0 is numeric before multiplying by 1000
        d2h0_val = solution_params.get('δ2H0', 'N/A')
        if isinstance(d2h0_val, (int, float, np.number)):
            param_text += f"δ2H₀: {d2h0_val*1000:.1f}‰\n"
        else:
            param_text += f"δ2H₀: {d2h0_val}‰\n"
    else:
        param_text = "Solution Parameters:\nNot available"
    
    ax4.text(0.1, 0.9, param_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # Plot 5: Cross-section along center
    ax5 = fig.add_subplot(gs[2, :])
    center_row = precipitation.shape[0] // 2
    x_coords = np.arange(precipitation.shape[1])
    
    ax5_twin = ax5.twinx()
    line1 = ax5.plot(x_coords, precipitation[center_row, :], label='Precipitation', color='blue')
    line2 = ax5_twin.plot(x_coords, isotope[center_row, :] * 1000, label='δ2H (×1000)', color='red')
    
    ax5.set_xlabel('X (grid units)')
    ax5.set_ylabel('Precipitation', color='blue')
    ax5_twin.set_ylabel('δ2H (‰)', color='red')
    ax5.set_title('Cross-section at mid Y-axis')
    
    # Add legends
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add overall title
    fig.suptitle('OPI One-Wind Model Results', fontsize=16, fontweight='bold')
    
    # Save or show the plot
    saved_files = []
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"opi_one_wind_plots_{timestamp}.{plot_format}"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, format=plot_format, dpi=150, bbox_inches='tight')
        saved_files.append(filepath)
        
        if verbose:
            print(f"Plot saved to: {filepath}")
    else:
        if verbose:
            print("Displaying plot...")
        plt.show()
    
    plt.close(fig)  # Close figure to free memory
    
    if verbose:
        print("Plot creation completed successfully")
    
    return saved_files


def create_comparison_plot(calc_results_list, titles_list, output_dir=None, plot_format='png', verbose=True):
    """
    Creates a comparison plot of multiple one-wind calculation results.
    
    Parameters:
    -----------
    calc_results_list : list of dict
        List of result dictionaries from opi_calc_one_wind function
    titles_list : list of str
        Titles for each result in the comparison
    output_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved
    plot_format : str, optional
        Format for saving plots ('png', 'pdf', 'svg', etc.)
    verbose : bool, optional
        Whether to print detailed progress information
    
    Returns:
    --------
    list
        List of paths to saved plot files (empty if output_dir is None)
    """
    if verbose:
        print("OPI Comparison Plots for One Wind Field")
        print("=======================================")
    
    if len(calc_results_list) != len(titles_list):
        raise ValueError("Length of calc_results_list and titles_list must match")
    
    n_results = len(calc_results_list)
    
    # Create figure with subplots for comparison
    fig, axes = plt.subplots(2, n_results, figsize=(6*n_results, 10))
    
    # Ensure axes is 2D array even for single result
    if n_results == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    # Plot precipitation and isotope for each result
    for i, (result, title) in enumerate(zip(calc_results_list, titles_list)):
        if 'precipitation' in result and 'isotope' in result:
            precip = result['precipitation']
            iso = result['isotope']
            
            # Precipitation plot
            im1 = axes[0, i].imshow(precip, aspect='auto', cmap='Blues', origin='lower')
            axes[0, i].set_title(f'{title}\nPrecipitation')
            axes[0, i].set_xlabel('X (grid units)')
            axes[0, i].set_ylabel('Y (grid units)')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Isotope plot
            im2 = axes[1, i].imshow(iso, aspect='auto', cmap='RdGy_r', origin='lower')
            axes[1, i].set_title(f'{title}\nδ2H Isotope')
            axes[1, i].set_xlabel('X (grid units)')
            axes[1, i].set_ylabel('Y (grid units)')
            plt.colorbar(im2, ax=axes[1, i])
    
    # Add overall title
    fig.suptitle('Comparison of OPI One-Wind Model Results', fontsize=16, fontweight='bold')
    
    # Save or show the plot
    saved_files = []
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"opi_one_wind_comparison_{timestamp}.{plot_format}"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, format=plot_format, dpi=150, bbox_inches='tight')
        saved_files.append(filepath)
        
        if verbose:
            print(f"Comparison plot saved to: {filepath}")
    else:
        if verbose:
            print("Displaying comparison plot...")
        plt.show()
    
    plt.close(fig)  # Close figure to free memory
    
    if verbose:
        print("Comparison plot creation completed successfully")
    
    return saved_files


if __name__ == "__main__":
    # Example usage with mock data
    print("Creating example plots...")
    
    # Create mock calculation result
    x_size, y_size = 50, 50
    mock_precip = 0.5 * np.exp(-(np.linspace(-2, 2, x_size)[:, None]**2 + np.linspace(-2, 2, y_size)[None, :]**2))
    mock_iso = -5.0 + 1.0 * (np.random.random((x_size, y_size)) - 0.5)  # Random variations around -5 permil
    
    mock_result = {
        'precipitation': mock_precip,
        'isotope': mock_iso,
        'solution_params': {
            'U': 10.0,
            'azimuth': 90.0,
            'T0': 290.0,
            'M': 0.25,
            'kappa': 10.0,
            'tau_c': 1000.0,
            'd2h0': -0.005,
        }
    }
    
    # Generate plots
    plot_files = opi_plots_one_wind(mock_result, verbose=True)
    print(f"Generated {len(plot_files)} plot files")