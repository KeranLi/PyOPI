"""
OPI Cross-Section Plotter

Generate cross-section plots (Fig 5 and 6) from MATLAB results.
This is a standalone script optimized for quick cross-section visualization.

Usage:
    python -m opi.cross-section <results_file> [options]
    
Examples:
    # Generate cross-sections with default settings
    python -m opi.cross-section ../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat
    
    # Specify output directory
    python -m opi.cross-section results.mat -o ./cross_sections
    
    # Adjust max grid size for memory control (lower = less memory)
    python -m opi.cross-section results.mat --max-grid-size 1000
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_matlab_results(results_file):
    """Load MATLAB results file, supporting both v7 and v7.3 formats."""
    try:
        return loadmat(results_file)
    except NotImplementedError:
        if not HAS_H5PY:
            raise ImportError("MATLAB v7.3 format detected but h5py is not installed.")
        
        results = {}
        with h5py.File(results_file, 'r') as f:
            def read_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data = np.array(obj)
                    if data.ndim >= 2:
                        data = data.T
                    results[name] = data
            
            f.visititems(read_dataset)
        
        return results


def extract_parameters(results):
    """Extract all needed parameters from results."""
    data = {}
    
    # Basic grids
    # NOTE: load_matlab_results already transposes h5py data to match MATLAB orientation
    # pGrid from h5py is already in (lat, lon) = (3560, 5748) order
    if 'pGrid' in results:
        data['p_grid'] = results['pGrid']
    if 'pGrid_1' in results:
        data['p_grid_1'] = results['pGrid_1']
    if 'pGrid_2' in results:
        data['p_grid_2'] = results['pGrid_2']
    
    if 'd2HGrid' in results:
        data['d2h_grid'] = results['d2HGrid']
    if 'd2HGrid_1' in results:
        data['d2h_grid_1'] = results['d2HGrid_1']
    if 'd2HGrid_2' in results:
        data['d2h_grid_2'] = results['d2HGrid_2']
    
    # Coordinates
    if 'lon' in results:
        data['lon'] = results['lon'].flatten()
    if 'lat' in results:
        data['lat'] = results['lat'].flatten()
    if 'x' in results:
        data['x'] = results['x'].flatten()
    if 'y' in results:
        data['y'] = results['y'].flatten()
    
    # Map origin and limits
    if 'lon0' in results:
        data['lon0'] = float(results['lon0'].flat[0])
    if 'lat0' in results:
        data['lat0'] = float(results['lat0'].flat[0])
    if 'mapLimits' in results:
        data['map_limits'] = results['mapLimits'].flatten()
    
    # Section origin
    if 'sectionLon0' in results:
        data['section_lon0'] = float(results['sectionLon0'].flat[0])
    if 'sectionLat0' in results:
        data['section_lat0'] = float(results['sectionLat0'].flat[0])
    
    # Wind parameters from beta
    if 'beta' in results:
        beta = results['beta'].flatten()
        if len(beta) >= 19:
            data['wind1_U'] = beta[0]
            data['azimuth_1'] = beta[1]
            data['m_1'] = beta[3]
            data['kappa_1'] = beta[4]
            data['tau_c_1'] = beta[5]
            data['d2h0_1'] = beta[6]
            data['dd2h0_dlat_1'] = beta[7]
            data['wind2_U'] = beta[10]
            data['azimuth_2'] = beta[11]
            data['m_2'] = beta[13]
            data['kappa_2'] = beta[14]
            data['tau_c_2'] = beta[15]
            data['d2h0_2'] = beta[16]
            data['dd2h0_dlat_2'] = beta[17]
    
    # Additional parameters
    for key, result_key in [
        ('h_s_1', 'hS_1'), ('h_s_2', 'hS_2'),
        ('h_rho_1', 'hRho_1'), ('h_rho_2', 'hRho_2'),
        ('rho_s0_1', 'rhoS0_1'), ('rho_s0_2', 'rhoS0_2'),
        ('tau_f_1', 'tauF_1'), ('tau_f_2', 'tauF_2'),
        ('gamma_ratio_1', 'gammaRatio_1'), ('gamma_ratio_2', 'gammaRatio_2'),
        ('f_c', 'fC'), ('h_r', 'hR')
    ]:
        if result_key in results:
            data[key] = float(results[result_key].flat[0])
    
    # Vertical profiles
    for key, result_key in [
        ('z_bar_1', 'zBar_1'), ('z_bar_2', 'zBar_2'),
        ('T_1', 'T_1'), ('T_2', 'T_2'),
        ('gamma_env_1', 'gammaEnv_1'), ('gamma_env_2', 'gammaEnv_2'),
        ('gamma_sat_1', 'gammaSat_1'), ('gamma_sat_2', 'gammaSat_2')
    ]:
        if result_key in results:
            data[key] = results[result_key].flatten()
    
    # Topography
    if 'hGrid' in results:
        data['h_grid'] = results['hGrid']
        print(f"  Found hGrid in results: shape={results['hGrid'].shape}")
    else:
        print(f"  hGrid not found in results. Available keys: {list(results.keys())[:20]}...")
    
    return data


def load_topography(data, results_file):
    """Load topography and coordinates from run file if not in results."""
    from .io.run_file import parse_run_file
    from .io.data_loader import get_input
    
    # Resolve to absolute path
    results_file = os.path.abspath(results_file)
    results_dir = os.path.dirname(results_file)
    
    # Search for any .run file in possible locations
    possible_dirs = [
        results_dir,
        os.path.join(results_dir, '..'),
        os.path.dirname(results_dir),
        os.path.join(os.path.dirname(results_dir), '..'),
    ]
    
    run_file_path = None
    for dir_path in possible_dirs:
        dir_path = os.path.abspath(dir_path)
        if os.path.isdir(dir_path):
            run_files = [f for f in os.listdir(dir_path) if f.endswith('.run')]
            if run_files:
                run_file_path = os.path.join(dir_path, run_files[0])
                break
    
    if run_file_path is None:
        print(f"  No .run file found in any of:")
        for dir_path in possible_dirs:
            print(f"    - {os.path.abspath(dir_path)}")
        return None, None, None, None, None
    
    print(f"  Found run file: {run_file_path}")
    
    try:
        run_params = parse_run_file(run_file_path)
        print(f"  Data path: {run_params.get('data_path', 'N/A')}")
        print(f"  Topo file: {run_params.get('topo_file', 'N/A')}")
        
        input_data = get_input(
            data_path=run_params['data_path'],
            topo_file=run_params['topo_file'],
            r_tukey=0.0,
            sample_file=None
        )
        return (
            input_data['h_grid'],
            input_data['lon'],
            input_data['lat'],
            input_data['x'],
            input_data['y']
        )
    except Exception as e:
        print(f"  Error loading topography: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def downsample_grid(h_grid, x, y, factor):
    """Downsample grid by given factor."""
    h_grid_ds = h_grid[::factor, ::factor]
    # Ensure coordinate vectors match the grid shape
    ny, nx = h_grid_ds.shape
    x_ds = x[::factor][:nx]  # Truncate to match grid
    y_ds = y[::factor][:ny]  # Truncate to match grid
    return h_grid_ds, x_ds, y_ds


def calculate_cross_section(data, state_num, max_grid_size=1500, verbose=True):
    """Calculate cross-section for given state."""
    from .wind_path import wind_path
    from .coordinates import lonlat2xy, xy2lonlat
    from .physics.cloud_water import calculate_cloud_water_section
    from .viz.cross_section import plot_cross_section
    from scipy.interpolate import RegularGridInterpolator
    import matplotlib.pyplot as plt
    
    suffix = f"_{state_num}"
    
    # Get parameters
    U = data.get(f'wind{state_num}_U', 10)
    azimuth = data.get(f'azimuth_{state_num}', 0)
    m = data.get(f'm_{state_num}', 1.0)
    kappa = data.get(f'kappa_{state_num}', 1000)
    tau_c = data.get(f'tau_c_{state_num}', 1000)
    tau_f = data.get(f'tau_f_{state_num}', 500)
    h_rho = data.get(f'h_rho_{state_num}', 8000)
    h_s = data.get(f'h_s_{state_num}', 1000)
    rho_s0 = data.get(f'rho_s0_{state_num}', 0.01)
    gamma_ratio = data.get(f'gamma_ratio_{state_num}', 0.8)
    d2h0 = data.get(f'd2h0_{state_num}', -100e-3)
    dd2h0_dlat = data.get(f'dd2h0_dlat_{state_num}', 0)
    
    z_bar = data.get(f'z_bar_{state_num}')
    T = data.get(f'T_{state_num}')
    gamma_env = data.get(f'gamma_env_{state_num}', 0.006)
    gamma_sat = data.get(f'gamma_sat_{state_num}', 0.005)
    
    # isotherm function expects arrays for gamma_env and gamma_sat
    # Keep them as arrays (they should be same length as z_bar)
    
    # Get coordinates
    # Note: In results file, x corresponds to h_grid[0] (rows) and y to h_grid[1] (columns)
    # This is different from the typical convention where x is columns and y is rows
    x, y = data['x'], data['y']
    lon, lat = data['lon'], data['lat']
    lon0, lat0 = data.get('lon0', np.mean(lon)), data.get('lat0', np.mean(lat))
    
    # Convert section origin
    section_lon0 = data.get('section_lon0', np.mean(lon))
    section_lat0 = data.get('section_lat0', np.mean(lat))
    x_section0, y_section0 = lonlat2xy(section_lon0, section_lat0, lon0, lat0)
    
    # Map limits
    map_limits = data.get('map_limits')
    if map_limits is not None and len(map_limits) >= 4:
        x_min, y_min = lonlat2xy(map_limits[0], map_limits[2], lon0, lat0)
        x_max, y_max = lonlat2xy(map_limits[1], map_limits[3], lon0, lat0)
        xy_limits = np.array([x_min, x_max, y_min, y_max])
    else:
        xy_limits = None
    
    # Downsample if needed
    # NOTE: wind_grid rotates the grid, which can increase dimensions by up to sqrt(2)
    # We need to be more conservative with max_grid_size to avoid memory issues
    ROTATION_FACTOR = 1.5  # Safety factor for grid rotation
    effective_max_size = int(max_grid_size / ROTATION_FACTOR)
    
    h_grid = data['h_grid']
    ny, nx = h_grid.shape
    
    # Save original coordinates for precipitation grid interpolation
    x_orig, y_orig = x, y
    
    # Check if x/y match h_grid dimensions correctly and swap if needed
    # We want x to match columns (nx) and y to match rows (ny)
    if len(x) == nx and len(y) == ny:
        # x matches columns, y matches rows - correct convention
        if verbose:
            print(f"  Coordinate check for h_grid: x matches columns ({nx}), y matches rows ({ny}) - correct")
    elif len(x) == ny and len(y) == nx:
        # x matches rows, y matches columns - need to swap
        if verbose:
            print(f"  Coordinate check for h_grid: swapping x and y (x was {len(x)}, y was {len(y)})")
        x, y = y, x
    
    if verbose:
        print(f"  Coordinates: x={len(x)}, y={len(y)}, h_grid={h_grid.shape}")
    
    factor = max(1, int(np.ceil(max(nx, ny) / effective_max_size)))
    
    if factor > 1:
        if verbose:
            print(f"  Down sampling grid by factor {factor} ({nx}x{ny} -> {nx//factor}x{ny//factor}, effective_max={effective_max_size})")
        h_grid_ds, x_ds, y_ds = downsample_grid(h_grid, x, y, factor)
        if verbose:
            print(f"  Downsampled: h_grid_ds={h_grid_ds.shape}, x_ds={len(x_ds)}, y_ds={len(y_ds)}")
    else:
        if verbose:
            print(f"  Grid size {nx}x{ny} within limits")
        h_grid_ds, x_ds, y_ds = h_grid, x, y
    
    # Ensure grid orientation matches coordinates
    # RegularGridInterpolator expects grid[i, j] where i corresponds to first coord, j to second
    # h_grid_ds should be (len(y_ds), len(x_ds))
    expected_shape = (len(y_ds), len(x_ds))
    if verbose:
        print(f"  Expected grid shape: {expected_shape}, actual: {h_grid_ds.shape}")
    if h_grid_ds.shape != expected_shape:
        if verbose:
            print(f"  Transposing grid to match coordinates")
        h_grid_ds = h_grid_ds.T
    
    # Calculate wind path
    if verbose:
        print(f"  Calculating wind path for state {state_num}...")
    x_path, y_path, s_path, s_limits = wind_path(
        x_section0, y_section0, azimuth, x_ds, y_ds, xy_limits
    )
    
    # Interpolate topography
    # h_grid_ds is (ny, nx) = (len(y_ds), len(x_ds)) but RegularGridInterpolator expects grid values in same order as coordinates
    # We need to transpose h_grid_ds to match (y, x) indexing
    try:
        h_interp = RegularGridInterpolator((y_ds, x_ds), h_grid_ds, method='linear', bounds_error=False, fill_value=0)
        h_l_path = h_interp(np.column_stack([y_path, x_path]))
    except ValueError as e:
        # If dimensions still don't match, try transposing
        if verbose:
            print(f"  Interpolation error, trying transposed: {e}")
        h_interp = RegularGridInterpolator((y_ds, x_ds), h_grid_ds.T, method='linear', bounds_error=False, fill_value=0)
        h_l_path = h_interp(np.column_stack([y_path, x_path]))
    
    # Calculate cloud water
    h_l_max = np.max(h_l_path) + 2 * h_s
    h_max = np.max(h_grid_ds)
    nm = m * U / h_max if h_max > 0 else 0.01
    f_c = data.get('f_c', 0)
    
    if verbose:
        print(f"  Calculating cloud water (this may take a moment)...")
    
    try:
        z_rho_c, rho_c, z_c_mean, z_248_path, z_268_path = calculate_cloud_water_section(
            x_path=x_path, y_path=y_path, z_max=h_l_max,
            x=x_ds, y=y_ds, h_grid=h_grid_ds,
            U=U, azimuth=azimuth, NM=nm, f_c=f_c,
            kappa=kappa, tau_c=tau_c, tau_f=tau_f, h_rho=h_rho,
            z_bar=z_bar, T=T,
            gamma_env=gamma_env, gamma_sat=gamma_sat,
            gamma_ratio=gamma_ratio, rho_s0=rho_s0, h_v=h_s
        )
    except MemoryError as e:
        print(f"  Error: Out of memory. Try increasing downsample factor with --max-grid-size")
        raise
    except ValueError as e:
        print(f"  ValueError in calculate_cloud_water_section: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Interpolate precipitation and d2H
    # NOTE: Use original projection coordinates (x_orig, y_orig) for precipitation grids
    # because pGrid and d2HGrid are defined on the original x, y grid from MATLAB results
    
    # If coordinates were swapped for h_grid, swap path coordinates back for p_grid
    if len(x_orig) == len(y) and len(y_orig) == len(x):
        y_path_p, x_path_p = x_path, y_path
    else:
        y_path_p, x_path_p = y_path, x_path
    
    p_interp = RegularGridInterpolator((y_orig, x_orig), data[f'p_grid_{state_num}'], 
                                       method='linear', bounds_error=False, fill_value=0)
    p_section = p_interp(np.column_stack([y_path_p, x_path_p]))
    
    d2h_interp = RegularGridInterpolator((y_orig, x_orig), data[f'd2h_grid_{state_num}'],
                                         method='linear', bounds_error=False, fill_value=0)
    d2h_section = d2h_interp(np.column_stack([y_path_p, x_path_p]))
    
    # Calculate base d2H using latitude
    lon_path, lat_path = xy2lonlat(x_path_p, y_path_p, lon0, lat0)
    d2h0_section = d2h0 + dd2h0_dlat * (lat_path - lat0)
    
    # Precipitation lines
    n_prec_lines = 20
    s_prec_offset = h_l_max * U * tau_f / h_s
    s_prec_lines = np.linspace(s_path[0] + s_prec_offset/2, s_path[-1] - s_prec_offset/2, n_prec_lines)
    s_prec_grid = np.array([s_prec_lines + s_prec_offset, s_prec_lines])
    h_prec_grid = np.array([np.zeros(n_prec_lines), h_l_max * np.ones(n_prec_lines)])
    
    return {
        's_path': s_path,
        'z_rho_c': z_rho_c,
        'rho_c': rho_c,
        'h_l_path': h_l_path,
        'z_248_path': z_248_path,
        'z_268_path': z_268_path,
        's_prec_lines': s_prec_grid,
        'h_prec_lines': h_prec_grid,
        'p_section': p_section,
        'd2h_section': d2h_section,
        'd2h0_section': d2h0_section,
        's_limits': s_limits,
        'h_l_max': h_l_max,
        'azimuth': azimuth
    }


def plot_cross_section_figure(cs_data, state_num, output_path=None, show=False):
    """Plot cross-section figure."""
    from .viz.cross_section import plot_cross_section
    import matplotlib.pyplot as plt
    
    title = f'Fig. {4 + state_num}. Cross Section for Precipitation State #{state_num}'
    
    fig, axes = plot_cross_section(
        s_path=cs_data['s_path'],
        z_rho_c=cs_data['z_rho_c'],
        rho_c=cs_data['rho_c'],
        h_l_path=cs_data['h_l_path'],
        z_248_path=cs_data['z_248_path'],
        z_268_path=cs_data['z_268_path'],
        s_l_section=None,
        z_l_section=None,
        s_prec_lines=cs_data['s_prec_lines'],
        h_prec_lines=cs_data['h_prec_lines'],
        p_section=cs_data['p_section'],
        d2h_section=cs_data['d2h_section'],
        d2h0_section=cs_data['d2h0_section'],
        s_limits=cs_data['s_limits'],
        h_l_max=cs_data['h_l_max'],
        azimuth=cs_data['azimuth'],
        title=title,
        save_path=output_path
    )
    
    if show:
        plt.show()
    
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate OPI cross-section plots from MATLAB results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.mat
  %(prog)s results.mat -o ./cross_sections
  %(prog)s results.mat --state 1 --max-grid-size 1000
        """
    )
    
    parser.add_argument('results_file', help='Path to opiCalc_TwoWinds_Results.mat')
    parser.add_argument('-o', '--output-dir', help='Output directory (default: same as results file)')
    parser.add_argument('--state', type=int, choices=[1, 2], help='Plot only specific state (1 or 2)')
    parser.add_argument('--max-grid-size', type=int, default=1500, 
                       help='Maximum grid size for FFT (default: 1500, lower = less memory)')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed progress')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: File not found: {args.results_file}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.results_file) or '.'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("OPI Cross-Section Plotter")
    print("=" * 40)
    print(f"Loading: {args.results_file}")
    
    # Load results
    try:
        results = load_matlab_results(args.results_file)
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    # Extract parameters
    data = extract_parameters(results)
    
    # Load topography if needed
    # NOTE: Keep coordinates from MATLAB results, only load h_grid from run file
    # because get_input coordinates may have different ordering than MATLAB results
    if 'h_grid' not in data or data['h_grid'] is None:
        print("Loading topography from run file...")
        h_grid, lon, lat, x, y = load_topography(data, args.results_file)
        if h_grid is None:
            print("Error: Could not load topography")
            sys.exit(1)
        data['h_grid'] = h_grid
        # Only use run file coordinates if MATLAB results don't have them
        if 'lon' not in data or data['lon'] is None:
            data['lon'] = lon
        if 'lat' not in data or data['lat'] is None:
            data['lat'] = lat
        if 'x' not in data or data['x'] is None:
            data['x'] = x
        if 'y' not in data or data['y'] is None:
            data['y'] = y
    
    # Check required data
    required = ['x', 'y', 'lon', 'lat', 'h_grid', 'p_grid_1', 'p_grid_2', 'd2h_grid_1', 'd2h_grid_2']
    missing = [r for r in required if r not in data]
    if missing:
        print(f"Error: Missing required data: {missing}")
        sys.exit(1)
    
    print(f"Grid size: {len(data['x'])} x {len(data['y'])}")
    print(f"Output: {output_dir}")
    print()
    
    # Determine which states to plot
    states = [args.state] if args.state else [1, 2]
    
    generated = []
    
    for state_num in states:
        print(f"Processing State {state_num}...")
        
        try:
            # Calculate cross-section data
            cs_data = calculate_cross_section(
                data, state_num, 
                max_grid_size=args.max_grid_size,
                verbose=args.verbose
            )
            
            # Plot figure
            output_path = os.path.join(output_dir, f'fig0{4+state_num}_cross_section_state{state_num}.png')
            plot_cross_section_figure(cs_data, state_num, output_path, show=args.show)
            
            print(f"  Saved: {output_path}")
            generated.append(output_path)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print(f"Generated {len(generated)} figure(s)")
    for path in generated:
        print(f"  {path}")


if __name__ == '__main__':
    main()
