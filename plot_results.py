"""
Plot OPI Simulation Results

Visualize the precipitation and isotope distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os

def plot_results(results_file='runs/demo_python/opiCalc_OneWind_Results.mat'):
    """Plot OPI simulation results"""
    
    # Load results
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        print("Please run a simulation first.")
        return
    
    data = loadmat(results_file)
    
    # Extract data
    h_grid = data['hGrid']
    p_grid = data['pGrid']
    d2h_grid = data['d2HGrid']
    d18o_grid = data['d18OGrid']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Topography
    ax = axes[0, 0]
    im = ax.imshow(h_grid, cmap='terrain', origin='lower')
    ax.set_title('Topography (m)')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    plt.colorbar(im, ax=ax)
    
    # 2. Precipitation
    ax = axes[0, 1]
    # Mask zero precipitation
    p_plot = p_grid.copy()
    p_plot[p_plot <= 0] = np.nan
    im = ax.imshow(p_plot * 1000, cmap='Blues', origin='lower')  # Convert to mm/s
    ax.set_title('Precipitation Rate (mm/s)')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    plt.colorbar(im, ax=ax)
    
    # 3. d2H
    ax = axes[1, 0]
    d2h_permil = d2h_grid * 1000
    # Clip extreme values for visualization
    vmin, vmax = np.percentile(d2h_permil[~np.isnan(d2h_permil)], [1, 99])
    im = ax.imshow(d2h_permil, cmap='Blues_r', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('Hydrogen Isotopes (d2H, permil)')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    plt.colorbar(im, ax=ax)
    
    # 4. d18O
    ax = axes[1, 1]
    d18o_permil = d18o_grid * 1000
    vmin, vmax = np.percentile(d18o_permil[~np.isnan(d18o_permil)], [1, 99])
    im = ax.imshow(d18o_permil, cmap='Blues_r', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('Oxygen Isotopes (d18O, permil)')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save figure
    output_file = results_file.replace('.mat', '_plot.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show statistics
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Topography: {h_grid.min():.0f} to {h_grid.max():.0f} m")
    print(f"Precipitation (non-zero): {p_grid[p_grid>0].min():.6f} to {p_grid.max():.6f} kg/m^2/s")
    print(f"d2H: {d2h_permil.min():.1f} to {d2h_permil.max():.1f} permil")
    print(f"d18O: {d18o_permil.min():.1f} to {d18o_permil.max():.1f} permil")
    
    # Calculate d-excess (excluding extreme values)
    d_excess = d2h_permil - 8 * d18o_permil
    valid_mask = (d_excess > -100) & (d_excess < 100)
    if valid_mask.any():
        print(f"d-excess (valid range): {d_excess[valid_mask].min():.1f} to {d_excess[valid_mask].max():.1f} permil")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot OPI Results')
    parser.add_argument('results_file', nargs='?', 
                       default='runs/demo_python/opiCalc_OneWind_Results.mat',
                       help='Path to results .mat file')
    
    args = parser.parse_args()
    
    plot_results(args.results_file)
