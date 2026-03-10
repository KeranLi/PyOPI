"""
One-Wind Visualization Function for OPI

This module creates map figures of OPI results for one-wind solutions.
Replicates the functionality of MATLAB's opiMaps_OneWind.m
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.io import loadmat
from datetime import datetime

from .io.run_file import parse_run_file
from .io.data_loader import grid_read
from .constants import M_PER_DEGREE, TC2K


def load_results(results_file, run_data=None):
    """
    Load results from OPI calculation.
    
    Parameters:
    -----------
    results_file : str
        Path to _Results.mat file
    run_data : dict, optional
        Run data dictionary if available
    
    Returns:
    --------
    dict
        Dictionary containing all result variables
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Load mat file
    mat_data = loadmat(results_file)
    
    # Extract variables
    results = {
        'run_path': str(mat_data.get('runPath', [''])[0]),
        'run_file': str(mat_data.get('runFile', [''])[0]),
        'run_title': str(mat_data.get('runTitle', [''])[0]),
        'data_path': str(mat_data.get('dataPath', [''])[0]),
        'topo_file': str(mat_data.get('topoFile', [''])[0]),
        'sample_file': str(mat_data.get('sampleFile', [''])[0]),
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
        'p_grid': mat_data['pGrid'],
        'd2h_grid': mat_data['d2HGrid'],
        'd18o_grid': mat_data['d18OGrid'],
        'f_m_grid': mat_data.get('fMGrid', np.ones_like(mat_data['pGrid'])),
    }
    
    # Optional variables
    if 'zBar' in mat_data:
        results['z_bar'] = float(mat_data['zBar'][0][0])
    if 'tauF' in mat_data:
        results['tau_f'] = float(mat_data['tauF'][0][0])
    
    # Sample data
    if 'sampleLon' in mat_data:
        results['sample_lon'] = mat_data['sampleLon'].flatten()
        results['sample_lat'] = mat_data['sampleLat'].flatten()
        results['sample_d2h'] = mat_data['sampleD2H'].flatten()
        results['sample_d18o'] = mat_data['sampleD18O'].flatten()
    
    return results


def create_topography_map(results, output_path=None, figsize=(12, 10)):
    """Create topography map."""
    fig, ax = plt.subplots(figsize=figsize)
    
    h_grid = results['h_grid']
    lon = results['lon']
    lat = results['lat']
    
    # Create hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(h_grid, vert_exag=0.1)
    
    # Plot topography
    im = ax.imshow(hillshade, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap='gray', origin='lower', alpha=0.5)
    
    # Overlay elevation contours
    contour = ax.contour(lon, lat, h_grid, levels=10, colors='brown', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Add samples if available
    if 'sample_lon' in results:
        scatter = ax.scatter(results['sample_lon'], results['sample_lat'],
                           c=results['sample_d2h'] * 1000, cmap='Blues_r',
                           edgecolors='black', s=50, zorder=5)
        plt.colorbar(scatter, ax=ax, label='d2H (permil)')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Topography and Sample Locations\n{results["run_title"]}')
    
    # Apply map limits if specified
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


def create_precipitation_map(results, output_path=None, figsize=(12, 10)):
    """Create precipitation rate map."""
    fig, ax = plt.subplots(figsize=figsize)
    
    p_grid = results['p_grid']
    lon = results['lon']
    lat = results['lat']
    
    # Mask zero precipitation
    p_plot = np.where(p_grid > 0, p_grid * 1000, np.nan)  # Convert to mm/s
    
    im = ax.imshow(p_plot, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                   cmap='YlGnBu', origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax, label='Precipitation Rate (mm/s)')
    
    # Add topography contours
    h_grid = results['h_grid']
    ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Precipitation Rate\n{results["run_title"]}')
    
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


def create_isotope_maps(results, output_path=None, figsize=(16, 6)):
    """Create d2H and d18O maps."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    d2h_grid = results['d2h_grid'] * 1000  # Convert to permil
    d18o_grid = results['d18o_grid'] * 1000
    lon = results['lon']
    lat = results['lat']
    h_grid = results['h_grid']
    
    # d2H map
    ax = axes[0]
    im1 = ax.imshow(d2h_grid, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                    cmap='Blues_r', origin='lower', aspect='auto',
                    vmin=np.nanpercentile(d2h_grid, 1),
                    vmax=np.nanpercentile(d2h_grid, 99))
    plt.colorbar(im1, ax=ax, label='d2H (permil)')
    ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Hydrogen Isotopes (d2H)')
    
    # d18O map
    ax = axes[1]
    im2 = ax.imshow(d18o_grid, extent=[lon[0], lon[-1], lat[0], lat[-1]],
                    cmap='Blues_r', origin='lower', aspect='auto',
                    vmin=np.nanpercentile(d18o_grid, 1),
                    vmax=np.nanpercentile(d18o_grid, 99))
    plt.colorbar(im2, ax=ax, label='d18O (permil)')
    ax.contour(lon, lat, h_grid, levels=5, colors='black', linewidths=0.5, alpha=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Oxygen Isotopes (d18O)')
    
    map_limits = results['map_limits']
    if not np.all(map_limits == 0):
        for ax in axes:
            ax.set_xlim(map_limits[0], map_limits[1])
            ax.set_ylim(map_limits[2], map_limits[3])
    
    plt.suptitle(f'Isotope Distribution\n{results["run_title"]}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axes


def create_observed_vs_predicted(results, output_path=None, figsize=(14, 6)):
    """Create observed vs predicted isotope plots."""
    if 'sample_lon' not in results:
        print("No sample data available for observed vs predicted plot")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    sample_d2h = results['sample_d2h'] * 1000
    sample_d18o = results['sample_d18o'] * 1000
    
    # Sample locations with observed values
    ax = axes[0]
    scatter = ax.scatter(results['sample_lon'], results['sample_lat'],
                        c=sample_d2h, cmap='Blues_r', edgecolors='black', s=100)
    plt.colorbar(scatter, ax=ax, label='Observed d2H (permil)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Observed d2H')
    
    ax = axes[1]
    scatter = ax.scatter(results['sample_lon'], results['sample_lat'],
                        c=sample_d18o, cmap='Blues_r', edgecolors='black', s=100)
    plt.colorbar(scatter, ax=ax, label='Observed d18O (permil)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Observed d18O')
    
    plt.suptitle(f'Observed Isotope Values\n{results["run_title"]}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axes


def create_parameter_summary(results, output_path=None, figsize=(10, 8)):
    """Create parameter summary table/figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    beta = results['beta']
    param_names = [
        'U (m/s)', 'Azimuth (deg)', 'T0 (K)', 'M (dimensionless)',
        'kappa (m²/s)', 'tau_c (s)', 'd2H0 (permil)', 
        'd(d2H0)/dLat (permil/deg)', 'fP0 (fraction)'
    ]
    
    # Create table data
    table_data = []
    for name, val in zip(param_names, beta):
        if 'd2H' in name or 'permil' in name:
            display_val = f"{val * 1000:.3f}"
        elif 'kappa' in name:
            display_val = f"{val:.2e}"
        else:
            display_val = f"{val:.4f}"
        table_data.append([name, display_val])
    
    # Add statistics
    table_data.append(['', ''])
    table_data.append(['Chi-square', f"{results.get('chi_r2', np.nan):.4f}"])
    if 'tau_f' in results:
        table_data.append(['Residence time (s)', f"{results['tau_f']:.1f}"])
    
    table = ax.table(cellText=table_data, loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(param_names)):
        table[(i, 0)].set_facecolor('#E6E6E6')
        table[(i, 1)].set_facecolor('#F0F0F0')
    
    ax.set_title(f'Best-Fit Parameters\n{results["run_title"]}', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, ax


def opi_maps_one_wind(results_file=None, run_file=None, output_dir=None,
                      save_plots=True, show_plots=False, verbose=True):
    """
    Create map figures of OPI one-wind results.
    
    Parameters:
    -----------
    results_file : str, optional
        Path to opiCalc_OneWind_Results.mat file
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
        print("OPI One-Wind Map Generation")
        print("=" * 40)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine results file
    if results_file is None:
        if run_file:
            run_data = parse_run_file(run_file)
            run_path = run_data['run_path']
            results_file = os.path.join(run_path, 'opiCalc_OneWind_Results.mat')
        else:
            raise ValueError("Either results_file or run_file must be provided")
    
    if verbose:
        print(f"Loading results from: {results_file}")
    
    results = load_results(results_file)
    
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
    
    # 1. Topography map
    if verbose:
        print("Creating topography map...")
    output_path = os.path.join(output_dir, f"{base_name}_topography.pdf") if save_plots else None
    create_topography_map(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    # 2. Precipitation map
    if verbose:
        print("Creating precipitation map...")
    output_path = os.path.join(output_dir, f"{base_name}_precipitation.pdf") if save_plots else None
    create_precipitation_map(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    # 3. Isotope maps
    if verbose:
        print("Creating isotope maps...")
    output_path = os.path.join(output_dir, f"{base_name}_isotopes.pdf") if save_plots else None
    create_isotope_maps(results, output_path)
    if output_path:
        generated_files.append(output_path)
    
    # 4. Observed vs predicted (if sample data available)
    if 'sample_lon' in results:
        if verbose:
            print("Creating observed values map...")
        output_path = os.path.join(output_dir, f"{base_name}_observed.pdf") if save_plots else None
        create_observed_vs_predicted(results, output_path)
        if output_path:
            generated_files.append(output_path)
    
    # 5. Parameter summary
    if verbose:
        print("Creating parameter summary...")
    output_path = os.path.join(output_dir, f"{base_name}_parameters.pdf") if save_plots else None
    create_parameter_summary(results, output_path)
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
        opi_maps_one_wind(results_file=sys.argv[1], save_plots=True, show_plots=False)
    else:
        print("Usage: python opi_maps_one_wind.py <results_file>")
