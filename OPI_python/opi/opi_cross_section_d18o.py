"""
OPI Cross-Section Plotter for d18O (Oxygen Isotopes)

Generate cross-section plots showing d18O instead of d2H.
Similar to Fig 5 and 6 but for oxygen isotopes.

Usage:
    python -m opi.cross-section-d18o <results_file> [options]
    
Examples:
    python -m opi.cross-section-d18o ../OPI_matlab/runs/opiCalc_TwoWinds_Results.mat
    python -m opi.cross-section-d18o results.mat -o ./d18o_sections --state 1
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
    
    # Debug: print available keys
    # print(f"  Available keys: {list(results.keys())[:20]}...")
    
    # Basic grids
    # NOTE: load_matlab_results already transposes h5py data to match MATLAB orientation
    # pGrid from h5py is already in (lat, lon) = (3560, 5748) order
    for key, result_key in [
        ('p_grid', 'pGrid'), ('p_grid_1', 'pGrid_1'), ('p_grid_2', 'pGrid_2'),
        ('d18o_grid', 'd18OGrid'), ('d18o_grid_1', 'd18OGrid_1'), ('d18o_grid_2', 'd18OGrid_2'),
        ('d2h_grid', 'd2HGrid'), ('d2h_grid_1', 'd2HGrid_1'), ('d2h_grid_2', 'd2HGrid_2')
    ]:
        if result_key in results:
            # Data already transposed by load_matlab_results
            data[key] = results[result_key]
    
    # Coordinates - check multiple possible key names
    # For coordinates, they might be stored differently in MATLAB v7.3
    coord_mappings = [
        ('lon', ['lon', 'Lon', 'LON', 'longitude', 'Longitude']),
        ('lat', ['lat', 'Lat', 'LAT', 'latitude', 'Latitude']),
        ('x', ['x', 'X']),
        ('y', ['y', 'Y'])
    ]
    
    for data_key, possible_keys in coord_mappings:
        for key in possible_keys:
            if key in results:
                data[data_key] = results[key].flatten()
                break
    
    # Map origin and limits (scalars)
    for key, result_key in [
        ('lon0', 'lon0'), ('lat0', 'lat0'),
        ('section_lon0', 'sectionLon0'), ('section_lat0', 'sectionLat0')
    ]:
        if result_key in results:
            val = results[result_key]
            data[key] = float(val.flat[0]) if hasattr(val, 'flat') else float(val)
    
    # Map limits (array)
    if 'mapLimits' in results:
        val = results['mapLimits']
        data['map_limits'] = val.flatten() if hasattr(val, 'flat') else np.array(val)
    
    # Wind parameters from beta
    if 'beta' in results:
        beta = results['beta'].flatten()
        if len(beta) >= 19:
            data['wind1_U'] = beta[0]
            data['azimuth_1'] = beta[1]
            data['m_1'] = beta[3]
            data['kappa_1'] = beta[4]
            data['tau_c_1'] = beta[5]
            data['d18o0_1'] = beta[6]  # Note: beta[6] is d2H0, need to calculate d18O0 from it
            data['dd18o0_dlat_1'] = beta[7]
            data['wind2_U'] = beta[10]
            data['azimuth_2'] = beta[11]
            data['m_2'] = beta[13]
            data['kappa_2'] = beta[14]
            data['tau_c_2'] = beta[15]
            data['d18o0_2'] = beta[16]
            data['dd18o0_dlat_2'] = beta[17]
    
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
    
    return data


def load_topography(data, results_file):
    """Load topography and coordinates from run file if not in results."""
    try:
        from ..io.run_file import parse_run_file
        from ..io.data_loader import get_input
    except ImportError:
        # When running as __main__, use absolute imports
        from opi.io.run_file import parse_run_file
        from opi.io.data_loader import get_input
    
    results_dir = os.path.dirname(os.path.abspath(results_file))
    
    # Search for any .run file
    possible_dirs = [results_dir, os.path.join(results_dir, '..'), 
                     os.path.dirname(results_dir)]
    
    run_file_path = None
    for dir_path in possible_dirs:
        dir_path = os.path.abspath(dir_path)
        if os.path.isdir(dir_path):
            run_files = [f for f in os.listdir(dir_path) if f.endswith('.run')]
            if run_files:
                run_file_path = os.path.join(dir_path, run_files[0])
                break
    
    if run_file_path is None:
        print(f"  No .run file found in: {possible_dirs}")
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


def calculate_cross_section_d18o(data, state_num, max_grid_size=1500, verbose=True):
    """Calculate cross-section for d18O."""
    try:
        from ..wind_path import wind_path
        from ..coordinates import lonlat2xy, xy2lonlat
        from ..physics.cloud_water import calculate_cloud_water_section
        from ..viz.cross_section import plot_cross_section
    except ImportError:
        # When running as __main__, use absolute imports
        from opi.wind_path import wind_path
        from opi.coordinates import lonlat2xy, xy2lonlat
        from opi.physics.cloud_water import calculate_cloud_water_section
        from opi.viz.cross_section import plot_cross_section
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
    
    # d18O parameters - convert from d2H if not available
    d18o0 = data.get(f'd18o0_{state_num}', data.get(f'd2h0_{state_num}', -100e-3) / 8)  # Approximate conversion
    dd18o0_dlat = data.get(f'dd18o0_dlat_{state_num}', data.get(f'dd2h0_dlat_{state_num}', 0) / 8)
    
    z_bar = data.get(f'z_bar_{state_num}')
    T = data.get(f'T_{state_num}')
    gamma_env = data.get(f'gamma_env_{state_num}')
    gamma_sat = data.get(f'gamma_sat_{state_num}')
    
    # If vertical profiles are missing, calculate them using base_state
    if z_bar is None or T is None:
        try:
            from ..physics.base_state import base_state
        except ImportError:
            from opi.physics.base_state import base_state
        
        # Get buoyancy frequency from m and U
        h_max = np.max(data['h_grid'])
        nm = m * U / h_max if h_max > 0 else 0.01
        
        # Assume surface temperature of 285K if not available
        T0 = 285.0
        
        if verbose:
            print(f"  Calculating vertical profiles (NM={nm:.4f}, T0={T0:.1f}K)...")
        
        z_bar, T, gamma_env, gamma_sat, _, _, _, _, _ = base_state(nm, T0)
        
        # gamma_ratio needs to be recalculated
        if gamma_ratio is None:
            gamma_ratio = 0.8  # Default value
    
    # Ensure arrays for gamma values
    if np.isscalar(gamma_env):
        gamma_env = np.full_like(z_bar, gamma_env) if z_bar is not None else np.array([0.006])
    if np.isscalar(gamma_sat):
        gamma_sat = np.full_like(z_bar, gamma_sat) if z_bar is not None else np.array([0.005])
    
    # Get coordinates
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
    factor = max(1, int(np.ceil(max(nx, ny) / effective_max_size)))
    
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
    
    if factor > 1:
        if verbose:
            print(f"  Down sampling grid by factor {factor} (max_grid_size={max_grid_size}, effective={effective_max_size})")
        h_grid_ds, x_ds, y_ds = downsample_grid(h_grid, x, y, factor)
    else:
        if verbose:
            print(f"  Grid size {nx}x{ny} within limits (max_grid_size={max_grid_size})")
        h_grid_ds, x_ds, y_ds = h_grid, x, y
    
    # Ensure grid orientation matches coordinates
    expected_shape = (len(y_ds), len(x_ds))
    if h_grid_ds.shape != expected_shape:
        h_grid_ds = h_grid_ds.T
    
    # Calculate wind path
    if verbose:
        print(f"  Calculating wind path...")
    x_path, y_path, s_path, s_limits = wind_path(
        x_section0, y_section0, azimuth, x_ds, y_ds, xy_limits
    )
    
    # Interpolate topography
    try:
        h_interp = RegularGridInterpolator((y_ds, x_ds), h_grid_ds, method='linear', bounds_error=False, fill_value=0)
    except ValueError:
        h_interp = RegularGridInterpolator((y_ds, x_ds), h_grid_ds.T, method='linear', bounds_error=False, fill_value=0)
    h_l_path = h_interp(np.column_stack([y_path, x_path]))
    
    # Calculate cloud water
    h_l_max = np.max(h_l_path) + 2 * h_s
    h_max = np.max(h_grid_ds)
    nm = m * U / h_max if h_max > 0 else 0.01
    f_c = data.get('f_c', 0)
    
    if verbose:
        print(f"  Calculating cloud water...")
    
    z_rho_c, rho_c, z_c_mean, z_248_path, z_268_path = calculate_cloud_water_section(
        x_path=x_path, y_path=y_path, z_max=h_l_max,
        x=x_ds, y=y_ds, h_grid=h_grid_ds,
        U=U, azimuth=azimuth, NM=nm, f_c=f_c,
        kappa=kappa, tau_c=tau_c, tau_f=tau_f, h_rho=h_rho,
        z_bar=z_bar, T=T,
        gamma_env=gamma_env, gamma_sat=gamma_sat,
        gamma_ratio=gamma_ratio, rho_s0=rho_s0, h_v=h_s
    )
    
    # Interpolate precipitation and d18O
    # NOTE: Use original projection coordinates (x_orig, y_orig) for precipitation grids
    # because pGrid and d18OGrid are defined on the original x, y grid from MATLAB results
    
    # If coordinates were swapped for h_grid, swap path coordinates back for p_grid
    if len(x_orig) == len(y) and len(y_orig) == len(x):
        y_path_p, x_path_p = x_path, y_path
    else:
        y_path_p, x_path_p = y_path, x_path
    
    p_interp = RegularGridInterpolator((y_orig, x_orig), data[f'p_grid_{state_num}'], 
                                       method='linear', bounds_error=False, fill_value=0)
    p_section = p_interp(np.column_stack([y_path_p, x_path_p]))
    
    # d18O interpolation
    d18o_interp = RegularGridInterpolator((y_orig, x_orig), data[f'd18o_grid_{state_num}'],
                                         method='linear', bounds_error=False, fill_value=0)
    d18o_section = d18o_interp(np.column_stack([y_path_p, x_path_p]))
    
    # Calculate base isotope values using latitude
    lon_path, lat_path = xy2lonlat(x_path_p, y_path_p, lon0, lat0)
    d18o0_section = d18o0 + dd18o0_dlat * (lat_path - lat0)
    
    # Also get d2H for comparison if available
    d2h_interp = RegularGridInterpolator((lat, lon), data[f'd2h_grid_{state_num}'],
                                         method='linear', bounds_error=False, fill_value=0)
    d2h_section = d2h_interp(np.column_stack([lat_path, lon_path]))
    
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
        'd18o_section': d18o_section,
        'd18o0_section': d18o0_section,
        'd2h_section': d2h_section,  # For comparison
        's_limits': s_limits,
        'h_l_max': h_l_max,
        'azimuth': azimuth
    }


def plot_cross_section_d18o(cs_data, state_num, output_path=None, show=False):
    """Plot cross-section figure with d18O panel."""
    try:
        from ..viz.cross_section import plot_cross_section
    except ImportError:
        # When running as __main__, use absolute imports
        from opi.viz.cross_section import plot_cross_section
    import matplotlib.pyplot as plt
    
    # Create custom figure with d18O instead of d2H
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Convert to km
    s_path_km = cs_data['s_path'] * 1e-3
    s_limits_km = (cs_data['s_limits'][0] * 1e-3, cs_data['s_limits'][1] * 1e-3)
    h_l_max_km = cs_data['h_l_max'] * 1e-3
    h_l_path_km = cs_data['h_l_path'] * 1e-3
    
    # Ensure z_rho_c is 1D
    z_rho_c = cs_data['z_rho_c']
    if z_rho_c.ndim > 1:
        z_rho_c = z_rho_c[:, 0]
    z_rho_c_km = z_rho_c * 1e-3
    
    # Top panel: Cloud water (simplified)
    rho_c = cs_data['rho_c']
    if rho_c is not None and rho_c.size > 0:
        im = ax1.pcolormesh(s_path_km, z_rho_c_km, rho_c, shading='auto', cmap='Greys')
        ax1.set_ylim([0, h_l_max_km])
        
        # Plot isotherms
        if cs_data['z_248_path'] is not None and cs_data['z_268_path'] is not None:
            ax1.plot(s_path_km, cs_data['z_248_path'] * 1e-3, 'b-', linewidth=1.5)
            ax1.plot(s_path_km, cs_data['z_268_path'] * 1e-3, 'b-', linewidth=1.5)
    
    # Topography
    topo_x = np.concatenate([[s_path_km[0]], s_path_km, [s_path_km[-1]]])
    topo_y = np.concatenate([[0], h_l_path_km, [0]])
    ax1.fill(topo_x, topo_y, color=[0.82, 0.70, 0.55], edgecolor='black', linewidth=0.5)
    
    # Precipitation lines
    if cs_data['s_prec_lines'] is not None and cs_data['h_prec_lines'] is not None:
        ax1.plot(cs_data['s_prec_lines'] * 1e-3, cs_data['h_prec_lines'] * 1e-3, 'k:', linewidth=1)
    
    ax1.set_ylabel('Elevation (km)', fontsize=10)
    ax1.set_title(f'Fig. {4+state_num}. Cross Section for Precipitation State #{state_num} (d18O)', fontsize=12)
    ax1.set_xlim(s_limits_km)
    ax1.tick_params(labelbottom=False)
    
    # Middle panel: Precipitation rate
    p_section = cs_data['p_section']
    if p_section is not None and p_section.size > 0:
        ax2.plot(s_path_km, p_section * 3.6e3, 'k-', linewidth=2)
        ax2.set_ylabel('Precipitation\nRate (mm/hr)', fontsize=10)
        ax2.set_ylim([0, np.max(p_section) * 1.05 * 3.6e3])
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelbottom=False)
        ax2.set_xlim(s_limits_km)
    
    # Bottom panel: d18O (the key difference!)
    d18o_section = cs_data['d18o_section']
    d18o0_section = cs_data['d18o0_section']
    if d18o_section is not None and d18o_section.size > 0:
        ax3.plot(s_path_km, d18o0_section * 1e3, '--k', linewidth=1, label='d18O0 (base)')
        ax3.plot(s_path_km, d18o_section * 1e3, '-k', linewidth=2, label='d18O (predicted)')
        ax3.set_ylabel('d18O (permil)', fontsize=10)
        ax3.set_xlabel('Section Distance (km)', fontsize=12)
        ax3.set_xlim(s_limits_km)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        # Set y-limits
        d18o_min = np.min(d18o_section) - 1e-3
        d18o_max = np.max(np.concatenate([d18o_section, [1e-3]]))
        ax3.set_ylim([d18o_min * 1e3, d18o_max * 1e3])
    
    # Wind direction arrow
    if ax2 is not None:
        ax2.annotate('', xy=(0.95, 0.5), xytext=(0.85, 0.5),
                    xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax2.text(0.9, 0.6, f'{cs_data["azimuth"]:.0f}°', transform=ax2.transAxes,
                fontsize=10, ha='center')
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.08, hspace=0.1)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate OPI cross-section plots for d18O (oxygen isotopes)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s results.mat
  %(prog)s results.mat -o ./d18o_sections
  %(prog)s results.mat --state 1 --max-grid-size 1000
        """
    )
    
    parser.add_argument('results_file', help='Path to opiCalc_TwoWinds_Results.mat')
    parser.add_argument('-o', '--output-dir', help='Output directory (default: same as results file)')
    parser.add_argument('--state', type=int, choices=[1, 2], help='Plot only specific state (1 or 2)')
    parser.add_argument('--max-grid-size', type=int, default=1500, 
                       help='Maximum grid size for FFT (default: 1500)')
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
    
    print("OPI Cross-Section Plotter for d18O")
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
    
    # Check if d18O data exists in results
    if 'd18o_grid_1' not in data or 'd18o_grid_2' not in data:
        print("\nWarning: d18O data not found in results file.")
        print("This results file may only contain d2H (hydrogen isotope) data.")
        print("\nOptions:")
        print("  1. Use the regular cross-section tool for d2H:")
        print(f"     python -m opi cross-section {args.results_file}")
        print("  2. Run the model with d18O calculation enabled")
        print("\nAlternatively, you can estimate d18O from d2H using the approximation d18O ≈ d2H / 8")
        response = input("\nWould you like to estimate d18O from d2H data? (y/n): ")
        if response.lower() == 'y':
            print("Estimating d18O from d2H using d18O = d2H / 8...")
            data['d18o_grid_1'] = data.get('d2h_grid_1', data['p_grid_1']) / 8.0
            data['d18o_grid_2'] = data.get('d2h_grid_2', data['p_grid_2']) / 8.0
            data['d18o_grid'] = data.get('d2h_grid', data['p_grid']) / 8.0
        else:
            sys.exit(1)
    
    # Check required data
    required = ['x', 'y', 'lon', 'lat', 'h_grid', 'p_grid_1', 'p_grid_2', 'd18o_grid_1', 'd18o_grid_2']
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
        print(f"Processing State {state_num} (d18O)...")
        
        try:
            # Calculate cross-section data
            cs_data = calculate_cross_section_d18o(
                data, state_num, 
                max_grid_size=args.max_grid_size,
                verbose=args.verbose
            )
            
            # Plot figure
            output_path = os.path.join(output_dir, f'fig0{4+state_num}_cross_section_d18o_state{state_num}.png')
            plot_cross_section_d18o(cs_data, state_num, output_path, show=args.show)
            
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
