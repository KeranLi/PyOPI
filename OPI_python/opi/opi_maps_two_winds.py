"""
Two-Wind Visualization Function for OPI

This module creates map figures of OPI results for two-wind solutions.
Replicates the functionality of MATLAB's opiMaps_TwoWinds.m
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.io import loadmat
from datetime import datetime

from .io.run_file import parse_run_file
from .constants import M_PER_DEGREE, TC2K


def load_results_two_winds(results_file):
    """Load results from two-wind OPI calculation."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    mat_data = loadmat(results_file)
    
    # Extract variables (similar to one-wind but with two states)
    results = {
        'run_path': str(mat_data.get('runPath', [''])[0]),
        'run_file': str(mat_data.get('runFile', [''])[0]),
        'run_title': str(mat_data.get('runTitle', [''])[0]),
        'data_path': str(mat_data.get('dataPath', [''])[0]),
        'topo_file': str(mat_data.get('topoFile', [''])[0]),
        'r_tukey': float(mat_data.get('rTukey', [[0]])[0][0]),
        'map_limits': mat_data.get('mapLimits', np.array([0, 0, 0, 0])).flatten(),
        'beta': mat_data['beta'].flatten(),
        'chi_r2': float(mat_data.get('chiR2', [[np.nan]])[0][0]),
        'lon': mat_data['lon'].flatten(),
        'lat': mat_data['lat'].flatten(),
        'x': mat_data['x'].flatten(),
        'y': mat_data['y'].flatten(),
        'lon0': float(mat_data['lon0'][0][0]),
        'lat0': float(mat_data['lat0'][0][0]),
        'h_grid': mat_data['hGrid'],
        # Combined results
        'p_grid': mat_data['pGrid'],
        'd2h_grid': mat_data['d2HGrid'],
        'd18o_grid': mat_data['d18OGrid'],
        # State 1 results
        'p_grid_1': mat_data.get('pGrid_1', mat_data['pGrid']),
        'd2h_grid_1': mat_data.get('d2HGrid_1', mat_data['d2HGrid']),
        'd18o_grid_1': mat_data.get('d18OGrid_1', mat_data['d18OGrid']),
        # State 2 results
        'p_grid_2': mat_data.get('pGrid_2', mat_data['pGrid']),
        'd2h_grid_2': mat_data.get('d2HGrid_2', mat_data['d2HGrid']),
        'd18o_grid_2': mat_data.get('d18OGrid_2', mat_data['d18OGrid']),
        # Fraction from state 1
        'fraction_p_grid': mat_data.get('fractionPGrid', 
                                        np.ones_like(mat_data['pGrid']) * 0.5),
    }
    
    # Sample data
    if 'sampleLon' in mat_data:
        results['sample_lon'] = mat_data['sampleLon'].flatten()
        results['sample_lat'] = mat_data['sampleLat'].flatten()
        results['sample_d2h'] = mat_data['sampleD2H'].flatten()
        results['sample_d18o'] = mat_data['sampleD18O'].flatten()
    
    # Sample side classification
    if 'isSampleSide01' in mat_data:
        results['is_sample_side_01'] = mat_data['isSampleSide01'].flatten().astype(bool)
    
    return results


def create_topography_map_two_winds(results, output_path=None, figsize=(12, 10)):
    """Create topography map with sample side classification."""
    fig, ax = plt.subplots(figsize=figsize)
    
    h_grid = results['h_grid']
    lon = results['lon']
    lat = results['lat']
    
    # Hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(h_grid, vert_exag=0.1)
    
    im = ax.imshow(hillshade, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap='gray', origin='lower', alpha=0.5)
    
    # Elevation contours
    contour = ax.contour(lon, lat, h_grid, levels=10, colors='brown', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Samples with side classification
    if 'sample_lon' in results:
        # Side 1 samples
        if 'is_sample_side_01' in results:
            side1 = results['is_sample_side_01']
            side2 = ~side1
            
            ax.scatter(results['sample_lon'][side1], results['sample_lat'][side1],
                      c='blue', marker='o', s=80, edgecolors='black',
                      label='Side 1 (Wind 1)', zorder=5)
            ax.scatter(results['sample_lon'][side2], results['sample_lat'][side2],
                      c='red', marker='s', s=80, edgecolors='black',
                      label='Side 2 (Wind 2)', zorder=5)
            ax.legend(loc='upper right')
        else:
            ax.scatter(results['sample_lon'], results['sample_lat'],
                      c='blue', edgecolors='black', s=50, zorder=5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Topography and Sample Classification\n{results["run_title"]}')
    
    map_limits = results['map_limits']
    if not np.all(map_limits == 0):
        ax.set_xlim(map_limits[0], map_limits[1])
        ax.set_ylim(map_limits[2], map_limits[3])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax


def create_precipitation_comparison(results, output_path=None, figsize=(18, 5)):
    """Create precipitation comparison maps (combined, state 1, state 2)."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    lon = results['lon']
    lat = results['lat']
    h_grid = results['h_grid']
    
    # Combined precipitation
    ax = axes[0]
    p_combined = np.where(results['p_grid'] > 0, results['p_grid'] * 1000, np.nan)
    im = ax.imshow(p_combined, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap='YlGnBu', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='Precipitation (mm/s)')
    ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
    ax.set_title('Combined Precipitation')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # State 1 precipitation
    ax = axes[1]
    p_1 = np.where(results['p_grid_1'] > 0, results['p_grid_1'] * 1000, np.nan)
    im = ax.imshow(p_1, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap='Blues', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='Precipitation (mm/s)')
    ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
    ax.set_title('Precipitation - Wind Field 1')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # State 2 precipitation
    ax = axes[2]
    p_2 = np.where(results['p_grid_2'] > 0, results['p_grid_2'] * 1000, np.nan)
    im = ax.imshow(p_2, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap='Oranges', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='Precipitation (mm/s)')
    ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
    ax.set_title('Precipitation - Wind Field 2')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    map_limits = results['map_limits']
    if not np.all(map_limits == 0):
        for ax in axes:
            ax.set_xlim(map_limits[0], map_limits[1])
            ax.set_ylim(map_limits[2], map_limits[3])
    
    plt.suptitle(f'Precipitation Comparison\n{results["run_title"]}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axes


def create_isotope_comparison(results, output_path=None, figsize=(18, 10)):
    """Create isotope comparison maps."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    lon = results['lon']
    lat = results['lat']
    h_grid = results['h_grid']
    
    d2h_combined = results['d2h_grid'] * 1000
    d2h_1 = results['d2h_grid_1'] * 1000
    d2h_2 = results['d2h_grid_2'] * 1000
    
    # d2H maps
    for i, (data, title, cmap) in enumerate([
        (d2h_combined, 'Combined d2H', 'Blues_r'),
        (d2h_1, 'd2H - Wind Field 1', 'Blues_r'),
        (d2h_2, 'd2H - Wind Field 2', 'Blues_r')
    ]):
        ax = axes[0, i]
        im = ax.imshow(data, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                      cmap=cmap, origin='lower', aspect='auto',
                      vmin=np.nanpercentile(d2h_combined, 1),
                      vmax=np.nanpercentile(d2h_combined, 99))
        plt.colorbar(im, ax=ax, label='d2H (permil)')
        ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    d18o_combined = results['d18o_grid'] * 1000
    d18o_1 = results['d18o_grid_1'] * 1000
    d18o_2 = results['d18o_grid_2'] * 1000
    
    # d18O maps
    for i, (data, title, cmap) in enumerate([
        (d18o_combined, 'Combined d18O', 'Blues_r'),
        (d18o_1, 'd18O - Wind Field 1', 'Blues_r'),
        (d18o_2, 'd18O - Wind Field 2', 'Blues_r')
    ]):
        ax = axes[1, i]
        im = ax.imshow(data, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                      cmap=cmap, origin='lower', aspect='auto',
                      vmin=np.nanpercentile(d18o_combined, 1),
                      vmax=np.nanpercentile(d18o_combined, 99))
        plt.colorbar(im, ax=ax, label='d18O (permil)')
        ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    map_limits = results['map_limits']
    if not np.all(map_limits == 0):
        for ax in axes.flat:
            ax.set_xlim(map_limits[0], map_limits[1])
            ax.set_ylim(map_limits[2], map_limits[3])
    
    plt.suptitle(f'Isotope Distribution Comparison\n{results["run_title"]}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axes


def create_fraction_map(results, output_path=None, figsize=(12, 10)):
    """Create map showing fraction of precipitation from wind field 1."""
    fig, ax = plt.subplots(figsize=figsize)
    
    lon = results['lon']
    lat = results['lat']
    h_grid = results['h_grid']
    fraction = results['fraction_p_grid']
    
    # Mask where no precipitation
    fraction_plot = np.where(results['p_grid'] > 0, fraction, np.nan)
    
    im = ax.imshow(fraction_plot, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap='RdYlBu_r', origin='lower', aspect='auto',
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction from Wind Field 1')
    
    # Topography contours
    ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Precipitation Source Fraction\n{results["run_title"]}')
    
    map_limits = results['map_limits']
    if not np.all(map_limits == 0):
        ax.set_xlim(map_limits[0], map_limits[1])
        ax.set_ylim(map_limits[2], map_limits[3])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax


def create_parameter_comparison(results, output_path=None, figsize=(14, 8)):
    """Create parameter comparison table for both wind fields."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    beta = results['beta']
    
    # Unpack parameters
    beta1 = beta[0:9]
    beta2 = beta[9:18]
    fraction = beta[9]
    
    param_names = [
        'U (m/s)', 'Azimuth (deg)', 'T0 (K)', 'M',
        'kappa (m²/s)', 'tau_c (s)', 'd2H0 (permil)',
        'd(d2H0)/dLat', 'fP0'
    ]
    
    # Wind field 1
    ax = axes[0]
    ax.axis('off')
    table_data = [['Parameter', 'Value']]
    for name, val in zip(param_names, beta1):
        if 'd2H' in name:
            display_val = f"{val * 1000:.2f}"
        elif 'kappa' in name:
            display_val = f"{val:.2e}"
        else:
            display_val = f"{val:.3f}"
        table_data.append([name, display_val])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Wind Field 1 Parameters', pad=20)
    
    # Wind field 2
    ax = axes[1]
    ax.axis('off')
    table_data = [['Parameter', 'Value']]
    for name, val in zip(param_names, beta2):
        if 'd2H' in name:
            display_val = f"{val * 1000:.2f}"
        elif 'kappa' in name:
            display_val = f"{val:.2e}"
        else:
            display_val = f"{val:.3f}"
        table_data.append([name, display_val])
    
    # Add fraction
    table_data.append(['', ''])
    table_data.append(['Fraction (Wind 1)', f'{fraction:.3f}'])
    table_data.append(['Chi-square', f'{results["chi_r2"]:.4f}'])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title('Wind Field 2 Parameters', pad=20)
    
    plt.suptitle(f'Best-Fit Parameters\n{results["run_title"]}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axes


def opi_maps_two_winds(results_file=None, run_file=None, output_dir=None,
                       save_plots=True, show_plots=False, verbose=True):
    """
    Create map figures of OPI two-wind results.
    
    Parameters:
    -----------
    results_file : str, optional
        Path to opiCalc_TwoWinds_Results.mat file
    run_file : str, optional
        Path to .run file (alternative to results_file)
    output_dir : str, optional
        Directory for output figures
    save_plots : bool, optional
        Whether to save plots to files
    show_plots : bool, optional
        Whether to display plots
    verbose : bool, optional
        Whether to print progress
    
    Returns:
    --------
    list
        List of generated figure paths
    """
    start_time = datetime.now()
    
    if verbose:
        print("OPI Two-Wind Map Generation")
        print("=" * 40)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine results file
    if results_file is None:
        if run_file:
            run_data = parse_run_file(run_file)
            run_path = run_data['run_path']
            results_file = os.path.join(run_path, 'opiCalc_TwoWinds_Results.mat')
        else:
            raise ValueError("Either results_file or run_file must be provided")
    
    if verbose:
        print(f"Loading results from: {results_file}")
    
    results = load_results_two_winds(results_file)
    
    if verbose:
        print(f"Run title: {results['run_title']}")
        print(f"Grid size: {results['h_grid'].shape}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    generated_files = []
    base_name = os.path.splitext(os.path.basename(results_file))[0]
    
    # 1. Topography with sample classification
    if verbose:
        print("Creating topography map...")
    output_path = os.path.join(output_dir, f"{base_name}_topography.pdf") if save_plots else None
    create_topography_map_two_winds(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    # 2. Precipitation comparison
    if verbose:
        print("Creating precipitation comparison...")
    output_path = os.path.join(output_dir, f"{base_name}_precipitation.pdf") if save_plots else None
    create_precipitation_comparison(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    # 3. Isotope comparison
    if verbose:
        print("Creating isotope comparison...")
    output_path = os.path.join(output_dir, f"{base_name}_isotopes.pdf") if save_plots else None
    create_isotope_comparison(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    # 4. Fraction map
    if verbose:
        print("Creating fraction map...")
    output_path = os.path.join(output_dir, f"{base_name}_fraction.pdf") if save_plots else None
    create_fraction_map(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    # 5. Parameter comparison
    if verbose:
        print("Creating parameter summary...")
    output_path = os.path.join(output_dir, f"{base_name}_parameters.pdf") if save_plots else None
    create_parameter_comparison(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    if verbose:
        print(f"\nGenerated {len(generated_files)} figures:")
        for f in generated_files:
            print(f"  {f}")
    
    if show_plots:
        plt.show()
    
    return generated_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        opi_maps_two_winds(results_file=sys.argv[1], save_plots=True, show_plots=False)
    else:
        print("Usage: python opi_maps_two_winds.py <results_file>")
