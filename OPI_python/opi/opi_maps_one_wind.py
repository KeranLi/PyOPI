"""
OPI Maps One Wind - Visualization for One-Wind Model Results

Equivalent to MATLAB's opiMaps_OneWind.m
"""

import os
import numpy as np
from scipy.io import loadmat

from .viz.maps import plot_result_maps
from .io.coordinates import xy2lonlat


def opi_maps_one_wind(results_file, output_dir=None, save_plots=True, 
                      show_plots=False, verbose=True):
    """
    Generate maps for one-wind OPI results.
    
    Parameters
    ----------
    results_file : str
        Path to opiCalc_OneWind_Results.mat file
    output_dir : str, optional
        Directory to save figures (default: same as results file)
    save_plots : bool, optional
        Whether to save plots to files (default: True)
    show_plots : bool, optional
        Whether to display plots interactively (default: False)
    verbose : bool, optional
        Whether to print progress information (default: True)
    
    Returns
    -------
    list
        List of generated figure file paths
    """
    if verbose:
        print("OPI Maps for One Wind Field")
        print("=" * 30)
        print(f"Loading results from: {results_file}")
    
    # Load results
    try:
        results = loadmat(results_file)
    except Exception as e:
        raise ValueError(f"Error loading results file: {e}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_file) or os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Output directory: {output_dir}")
    
    # Prepare data dictionary
    data = _prepare_data_dict(results)
    
    # Generate maps
    generated_files = []
    
    if save_plots:
        if verbose:
            print("\nGenerating maps...")
        
        # Use the viz module function
        files = plot_result_maps(data, two_wind=False, save_dir=output_dir)
        generated_files.extend(files)
        
        if verbose:
            print(f"Generated {len(files)} maps")
    
    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()
    
    if verbose:
        print(f"\nTotal figures generated: {len(generated_files)}")
    
    return generated_files


def _prepare_data_dict(results):
    """Prepare data dictionary from loaded MATLAB results."""
    data = {}
    
    # Extract grids - use keys that plot_result_maps expects
    if 'pGrid' in results:
        data['p_grid'] = results['pGrid']
    if 'd2HGrid' in results:
        data['d2h_grid'] = results['d2HGrid']
    if 'd18OGrid' in results:
        data['d18o_grid'] = results['d18OGrid']
    
    # Get coordinate grids
    if 'lon' in results:
        data['lon'] = results['lon'].flatten()
        data['lat'] = results['lat'].flatten() if 'lat' in results else None
    elif 'x' in results:
        data['x'] = results['x'].flatten()
        data['y'] = results['y'].flatten() if 'y' in results else None
        data['lon'] = data['x']
        data['lat'] = data['y']
    else:
        # Try to load from run file
        data['lon'], data['lat'] = _load_coordinates_from_run(results, results_file)
    
    if 'hGrid' in results:
        data['h_grid'] = results['hGrid']
    
    # Get parameters if available
    if 'beta' in results:
        data['beta'] = results['beta'].flatten()
    
    return data


def _load_coordinates_from_run(results, results_file):
    """Try to load coordinates from run file or data directory."""
    try:
        from .io.run_file import parse_run_file
        from .io.data_loader import get_input
        
        run_path = results.get('runPath', [os.path.dirname(results_file)])[0]
        run_file = results.get('runFile', ['run.run'])[0]
        run_file_path = os.path.join(run_path, run_file)
        
        if os.path.exists(run_file_path):
            run_params = parse_run_file(run_file_path)
            input_data = get_input(
                data_path=run_params['data_path'],
                topo_file=run_params['topo_file'],
                r_tukey=0.0,
                sample_file=None
            )
            return input_data['lon'], input_data['lat']
    except Exception:
        pass
    
    # Fallback: create synthetic coordinates
    if 'pGrid' in results:
        shape = results['pGrid'].shape
        lon = np.linspace(0, 1, shape[1])
        lat = np.linspace(0, 1, shape[0])
        return lon, lat
    
    return None, None


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python opi_maps_one_wind.py <results_file> [output_dir]")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    files = opi_maps_one_wind(results_file, output_dir, verbose=True)
    print(f"Generated files: {files}")
