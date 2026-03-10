"""
Main OPI Simulation Runner

This module provides the main entry point for running OPI simulations
from .run configuration files, matching the MATLAB workflow:
1. Parse .run file
2. Load input data (topography and samples)
3. Run calculation (one-wind or two-wind)
4. Save results to .mat file
5. Generate log file

Usage:
    python -m opi.run_opi_simulation path/to/config.run
    
Or from Python:
    from opi.run_opi_simulation import run_simulation
    results = run_simulation('path/to/config.run')
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime
from scipy.io import savemat

# Import OPI modules
from .io.run_file import parse_run_file
from .io.data_loader import get_input
from .calc_one_wind import calc_one_wind
from .calc_two_winds import calc_two_winds
from .catchment_nodes import catchment_nodes
from .coordinates import lonlat2xy
from .constants import SD_RES_RATIO, HR, TC2K, RADIUS_EARTH, M_PER_DEGREE


def setup_logging(log_path, run_title, run_file):
    """Set up log file and return log function."""
    log_lines = []
    
    def log(message, to_console=True):
        """Log message to file and optionally console."""
        log_lines.append(message)
        if to_console:
            print(message)
    
    # Initial log entries
    start_time = datetime.now()
    log(f"Program: OPI Simulation Runner")
    log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Run file: {run_file}")
    log(f"Run title: {run_title}")
    log("")
    
    return log, log_lines, start_time


def save_log(log_path, log_lines):
    """Save log to file."""
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))


def determine_model_type(run_data):
    """Determine if this is one-wind or two-wind model."""
    n_params = run_data['n_parameters']
    
    if n_params == 9:
        return 'one_wind'
    elif n_params == 19:
        return 'two_winds'
    else:
        raise ValueError(f"Invalid number of parameters: {n_params}. Expected 9 or 19.")


def load_continental_divide(data_path, divide_file, log):
    """Load continental divide data for two-wind model."""
    if not divide_file:
        return None, None
    
    divide_path = os.path.join(data_path, divide_file)
    if not os.path.exists(divide_path):
        log(f"Warning: Continental divide file not found: {divide_path}")
        return None, None
    
    try:
        from scipy.io import loadmat
        divide_data = loadmat(divide_path)
        cont_divide_lon = divide_data['contDivideLon'].flatten()
        cont_divide_lat = divide_data['contDivideLat'].flatten()
        log(f"Loaded continental divide: {len(cont_divide_lon)} points")
        return cont_divide_lon, cont_divide_lat
    except Exception as e:
        log(f"Warning: Failed to load continental divide: {e}")
        return None, None


def determine_sample_sides(sample_x, sample_y, cont_divide_x, cont_divide_y, 
                           azimuth_1, region_x, region_y, lon0, lat0):
    """
    Determine which side of continental divide each sample falls on.
    
    Returns boolean array: True = side 1, False = side 2
    """
    from matplotlib.path import Path
    
    # Find intersections between divide and region boundary
    # Simplified: use polygon containment test
    
    # Create a polygon representing the region side for wind 1
    # This is a simplified version - full implementation would use polyxpoly
    
    # Calculate which side of the divide the wind enters from
    mean_x = np.mean(region_x)
    mean_y = np.mean(region_y)
    r = min([region_x.max() - region_x.min(), region_y.max() - region_y.min()]) / 2
    
    # Wind entry point
    wind_entry_x = mean_x + r * np.sin(np.deg2rad(-azimuth_1))
    wind_entry_y = mean_y + r * np.cos(np.deg2rad(-azimuth_1))
    
    # Determine which side of divide the wind comes from
    # Simplified: assume divide roughly separates the two sides
    
    # For now, use simple distance-based classification
    # This should be replaced with proper polygon intersection logic
    n_samples = len(sample_x)
    
    # Calculate distance from each sample to divide line
    # Samples on one side belong to wind 1, the other side to wind 2
    
    # Create a simple line from divide endpoints
    if len(cont_divide_x) >= 2:
        # Use line equation: ax + by + c = 0
        x1, y1 = cont_divide_x[0], cont_divide_y[0]
        x2, y2 = cont_divide_x[-1], cont_divide_y[-1]
        
        # Normal vector pointing to one side
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        
        # Determine side for each sample
        distances = a * sample_x + b * sample_y + c
        
        # Determine which side is wind 1
        wind_side = a * wind_entry_x + b * wind_entry_y + c
        
        # Samples on same side as wind entry are for wind 1
        is_side_1 = (distances * wind_side) > 0
        
        return is_side_1
    
    # Fallback: alternate assignment
    return np.arange(n_samples) < n_samples // 2


def run_one_wind_simulation(run_data, input_data, log):
    """Run one-wind simulation."""
    log("\n" + "="*50)
    log("RUNNING ONE-WIND SIMULATION")
    log("="*50)
    
    # Get solution vector
    if run_data.get('beta') is not None:
        beta = np.array(run_data['beta'])
        log(f"Using solution vector from run file")
    else:
        raise ValueError("One-wind simulation requires a solution vector (beta) in run file")
    
    if len(beta) != 9:
        raise ValueError(f"Expected 9 parameters for one-wind, got {len(beta)}")
    
    # Unpack parameters for logging
    param_names = ['U', 'Azimuth', 'T0', 'M', 'kappa', 'tau_c', 'd2H0', 'd_d2H0_d_lat', 'fP0']
    log("\nSolution parameters:")
    for name, val in zip(param_names, beta):
        log(f"  {name}: {val}")
    
    # Calculate catchment nodes
    sample_lc = input_data.get('sample_lc')
    if sample_lc is None or len(sample_lc) == 0:
        ij_catch = []
        ptr_catch = [0]
        n_samples = 0
    else:
        ij_catch, ptr_catch = catchment_nodes(
            input_data['sample_x'], input_data['sample_y'],
            sample_lc, input_data['x'], input_data['y'],
            input_data['h_grid']
        )
        n_samples = len(input_data['sample_x'])
    
    log(f"\nNumber of samples: {n_samples}")
    
    # Prepare sample data
    if n_samples > 0:
        sample_d2h = input_data['sample_d2h']
        sample_d18o = input_data['sample_d18o']
    else:
        sample_d2h = np.array([np.nan])
        sample_d18o = np.array([np.nan])
    
    # Prepare covariance
    cov = input_data.get('cov', np.array([[1e-6, 0], [0, 1e-6]]))
    
    # Number of free parameters
    n_free = run_data['n_free']
    
    # Run calculation
    log("\nRunning calculation...")
    result = calc_one_wind(
        beta=beta,
        f_c=input_data['f_c'],
        h_r=input_data.get('h_r', HR),
        x=input_data['x'],
        y=input_data['y'],
        lat=np.array([input_data['lat0']]),
        lat0=input_data['lat0'],
        h_grid=input_data['h_grid'],
        b_mwl_sample=input_data['b_mwl_sample'],
        ij_catch=ij_catch,
        ptr_catch=ptr_catch,
        sample_d2h=sample_d2h,
        sample_d18o=sample_d18o,
        cov=cov,
        n_parameters_free=n_free,
        is_fit=False
    )
    
    # Unpack results
    (chi_r2, nu, std_residuals, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
     rho_s0, h_s, rho0, h_rho, d18o0, d_d18o0_d_lat, tau_f, p_grid, f_m_grid, r_h_grid,
     evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid,
     d2h_grid, d18o_grid, i_wet, d2h_pred, d18o_pred) = result
    
    log("\nCalculation completed!")
    
    # Log results
    U = beta[0]
    azimuth = beta[1]
    T0 = beta[2]
    M = beta[3]
    h_max = np.max(input_data['h_grid'])
    NM = M * U / h_max
    
    log(f"\nResults:")
    log(f"  Moist buoyancy frequency: {NM*1000:.3f} mrad/s")
    log(f"  Water vapor scale height: {h_s:.0f} m")
    log(f"  Falling precipitation residence time: {tau_f:.0f} s")
    log(f"  Mean precipitation: {np.mean(p_grid):.6f} kg/m^2/s")
    log(f"  d2H range: {np.min(d2h_grid)*1000:.1f} to {np.max(d2h_grid)*1000:.1f} permil")
    log(f"  d18O range: {np.min(d18o_grid)*1000:.1f} to {np.max(d18o_grid)*1000:.1f} permil")
    
    if not np.isnan(chi_r2):
        log(f"  Reduced chi-square: {chi_r2:.4f}")
        log(f"  Degrees of freedom: {nu}")
    
    return {
        'beta': beta,
        'chi_r2': chi_r2,
        'nu': nu,
        'p_grid': p_grid,
        'd2h_grid': d2h_grid,
        'd18o_grid': d18o_grid,
        'f_m_grid': f_m_grid,
        'z_bar': z_bar, 'T': T, 
        'gamma_env': gamma_env, 'gamma_sat': gamma_sat, 'gamma_ratio': gamma_ratio,
        'rho_s0': rho_s0, 'h_s': h_s, 'rho0': rho0, 'h_rho': h_rho,
        'tau_f': tau_f,
        'd2h_pred': d2h_pred, 'd18o_pred': d18o_pred,
        'ij_catch': ij_catch, 'ptr_catch': ptr_catch
    }


def run_two_winds_simulation(run_data, input_data, log):
    """Run two-winds simulation."""
    log("\n" + "="*50)
    log("RUNNING TWO-WINDS SIMULATION")
    log("="*50)
    
    # Get solution vector
    if run_data.get('beta') is not None:
        beta = np.array(run_data['beta'])
        log(f"Using solution vector from run file")
    else:
        raise ValueError("Two-winds simulation requires a solution vector (beta) in run file")
    
    if len(beta) != 19:
        raise ValueError(f"Expected 19 parameters for two-winds, got {len(beta)}")
    
    # Unpack parameters
    beta1 = beta[0:9]
    beta2 = beta[9:18]
    fraction = beta[9]  # fraction of wind 1 (parameter 10)
    
    param_names = ['U', 'Azimuth', 'T0', 'M', 'kappa', 'tau_c', 'd2H0', 'd_d2H0_d_lat', 'fP0']
    
    log("\nWind Field 1 Parameters:")
    for name, val in zip(param_names, beta1):
        log(f"  {name}: {val}")
    
    log("\nWind Field 2 Parameters:")
    for name, val in zip(param_names, beta2):
        log(f"  {name}: {val}")
    
    log(f"\nFraction (Wind 1): {fraction:.2f}")
    
    # Load continental divide if available
    data_path = run_data['data_path']
    divide_file = run_data.get('cont_divide_file')
    cont_divide_lon, cont_divide_lat = load_continental_divide(data_path, divide_file, log)
    
    # Calculate catchment nodes
    sample_lc = input_data.get('sample_lc')
    if sample_lc is None or len(sample_lc) == 0:
        ij_catch = []
        ptr_catch = [0]
        n_samples = 0
        is_sample_side_1 = np.array([], dtype=bool)
    else:
        ij_catch, ptr_catch = catchment_nodes(
            input_data['sample_x'], input_data['sample_y'],
            sample_lc, input_data['x'], input_data['y'],
            input_data['h_grid']
        )
        n_samples = len(input_data['sample_x'])
        
        # Determine which side each sample is on
        if cont_divide_lon is not None:
            # Convert divide to xy coordinates
            cont_divide_x, cont_divide_y = lonlat2xy(
                cont_divide_lon, cont_divide_lat,
                input_data['lon0'], input_data['lat0']
            )
            
            # Create region boundary
            x, y = input_data['x'], input_data['y']
            region_x = np.array([x[0], x[-1], x[-1], x[0], x[0]])
            region_y = np.array([y[0], y[0], y[-1], y[-1], y[0]])
            
            # Determine sample sides
            is_sample_side_1 = determine_sample_sides(
                input_data['sample_x'], input_data['sample_y'],
                cont_divide_x, cont_divide_y,
                beta1[1],  # azimuth of wind 1
                region_x, region_y,
                input_data['lon0'], input_data['lat0']
            )
        else:
            # Default: all samples use combined prediction
            is_sample_side_1 = np.ones(n_samples, dtype=bool)
    
    log(f"\nNumber of samples: {n_samples}")
    if n_samples > 0:
        log(f"  Side 1 (Wind 1): {np.sum(is_sample_side_1)}")
        log(f"  Side 2 (Wind 2): {np.sum(~is_sample_side_1)}")
    
    # Prepare sample data
    if n_samples > 0:
        sample_d2h = input_data['sample_d2h']
        sample_d18o = input_data['sample_d18o']
    else:
        sample_d2h = np.array([np.nan])
        sample_d18o = np.array([np.nan])
    
    # Prepare covariance
    cov = input_data.get('cov', np.array([[1e-6, 0], [0, 1e-6]]))
    
    # Number of free parameters
    n_free = run_data['n_free']
    
    # Run calculation
    log("\nRunning calculation...")
    result = calc_two_winds(
        beta=beta,
        f_c=input_data['f_c'],
        h_r=input_data.get('h_r', HR),
        x=input_data['x'],
        y=input_data['y'],
        lat=np.array([input_data['lat0']]),
        lat0=input_data['lat0'],
        h_grid=input_data['h_grid'],
        b_mwl_sample=input_data['b_mwl_sample'],
        ij_catch=ij_catch,
        ptr_catch=ptr_catch,
        sample_d2h=sample_d2h,
        sample_d18o=sample_d18o,
        cov=cov,
        n_parameters_free=n_free,
        is_fit=False
    )
    
    # Unpack results (two-winds returns more values)
    (chi_r2, nu, std_residuals, 
     z_bar_1, T_1, gamma_env_1, gamma_sat_1, gamma_ratio_1,
     rho_s0_1, h_s_1, rho0_1, h_rho_1,
     d18o0_1, d_d18o0_d_lat_1, tau_f_1, p_grid_1, f_m_grid_1, r_h_grid_1,
     d2h_grid_1, d18o_grid_1, p_sum_pred_1, d2h_pred_1, d18o_pred_1,
     z_bar_2, T_2, gamma_env_2, gamma_sat_2, gamma_ratio_2,
     rho_s0_2, h_s_2, rho0_2, h_rho_2,
     d18o0_2, d_d18o0_d_lat_2, tau_f_2, p_grid_2, f_m_grid_2, r_h_grid_2,
     d2h_grid_2, d18o_grid_2, p_sum_pred_2, d2h_pred_2, d18o_pred_2,
     p_grid, d2h_grid, d18o_grid, p_sum_pred, d2h_pred, d18o_pred,
     fraction_p_grid) = result
    
    log("\nCalculation completed!")
    
    # Log results
    log(f"\nResults:")
    log(f"  Wind 1 - Mean precipitation: {np.mean(p_grid_1):.6f} kg/m^2/s")
    log(f"  Wind 2 - Mean precipitation: {np.mean(p_grid_2):.6f} kg/m^2/s")
    log(f"  Combined - Mean precipitation: {np.mean(p_grid):.6f} kg/m^2/s")
    log(f"  d2H range: {np.min(d2h_grid)*1000:.1f} to {np.max(d2h_grid)*1000:.1f} permil")
    log(f"  d18O range: {np.min(d18o_grid)*1000:.1f} to {np.max(d18o_grid)*1000:.1f} permil")
    
    if not np.isnan(chi_r2):
        log(f"  Reduced chi-square: {chi_r2:.4f}")
        log(f"  Degrees of freedom: {nu}")
    
    return {
        'beta': beta,
        'chi_r2': chi_r2,
        'nu': nu,
        'p_grid': p_grid,
        'd2h_grid': d2h_grid,
        'd18o_grid': d18o_grid,
        'p_grid_1': p_grid_1, 'p_grid_2': p_grid_2,
        'd2h_grid_1': d2h_grid_1, 'd2h_grid_2': d2h_grid_2,
        'd18o_grid_1': d18o_grid_1, 'd18o_grid_2': d18o_grid_2,
        'fraction_p_grid': fraction_p_grid,
        'is_sample_side_1': is_sample_side_1,
        'ij_catch': ij_catch, 'ptr_catch': ptr_catch
    }


def run_simulation(run_file_path, verbose=True):
    """
    Run OPI simulation from a .run configuration file.
    
    Parameters:
    -----------
    run_file_path : str
        Path to the .run configuration file
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict
        Dictionary containing simulation results and output paths
    """
    # Parse run file
    if verbose:
        print(f"Loading run file: {run_file_path}")
    
    run_data = parse_run_file(run_file_path)
    run_path = run_data['run_path']
    run_filename = run_data['run_file']
    
    # Set up logging
    log_filename = os.path.join(run_path, f"opi_simulation_log.txt")
    log, log_lines, start_time = setup_logging(log_filename, run_data['run_title'], run_file_path)
    
    try:
        # Determine model type
        model_type = determine_model_type(run_data)
        log(f"Model type: {model_type}")
        
        # Load input data
        data_path = run_data['data_path']
        log(f"Data path: {data_path}")
        
        input_data = get_input(
            data_path=data_path,
            topo_file=run_data['topo_file'],
            r_tukey=run_data['r_tukey'],
            sample_file=run_data.get('sample_file'),
            sd_res_ratio=SD_RES_RATIO
        )
        
        h_grid = input_data['h_grid']
        log(f"Topography loaded: {h_grid.shape[1]} x {h_grid.shape[0]} grid")
        log(f"Maximum elevation: {np.max(h_grid):.0f} m")
        log(f"Grid extent: {input_data['x'][0]/1000:.1f} to {input_data['x'][-1]/1000:.1f} km (x)")
        log(f"             {input_data['y'][0]/1000:.1f} to {input_data['y'][-1]/1000:.1f} km (y)")
        
        # Run simulation
        if model_type == 'one_wind':
            results = run_one_wind_simulation(run_data, input_data, log)
            results_filename = f"opiCalc_OneWind_Results.mat"
        else:
            results = run_two_winds_simulation(run_data, input_data, log)
            results_filename = f"opiCalc_TwoWinds_Results.mat"
        
        # Save results
        results_path = os.path.join(run_path, results_filename)
        
        # Prepare save data (matching MATLAB format as closely as possible)
        save_data = {
            'startTimeOpiCalc': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'runPath': run_path,
            'runFile': run_filename,
            'runTitle': run_data['run_title'],
            'dataPath': data_path,
            'topoFile': run_data['topo_file'],
            'rTukey': run_data['r_tukey'],
            'sampleFile': run_data.get('sample_file', ''),
            'contDivideFile': run_data.get('cont_divide_file', ''),
            'mapLimits': np.array(run_data['map_limits']),
            'sectionLon0': run_data.get('section_lon0') if run_data.get('section_lon0') is not None else 0,
            'sectionLat0': run_data.get('section_lat0') if run_data.get('section_lat0') is not None else 0,
            'lon': input_data['lon'],
            'lat': input_data['lat'],
            'x': input_data['x'],
            'y': input_data['y'],
            'lon0': input_data['lon0'],
            'lat0': input_data['lat0'],
            'hGrid': h_grid,
            'beta': results['beta'],
            'chiR2': results['chi_r2'],
            'nu': results['nu'],
            'pGrid': results['p_grid'],
            'd2HGrid': results['d2h_grid'],
            'd18OGrid': results['d18o_grid'],
            'lB': np.array(run_data['l_b']),
            'uB': np.array(run_data['u_b']),
            'nParametersFree': run_data['n_free'],
            'fC': input_data['f_c'],
            'hR': input_data.get('h_r', HR),
            'sdResRatio': SD_RES_RATIO,
        }
        
        # Add model-specific outputs
        if model_type == 'two_winds':
            save_data.update({
                'pGrid_1': results['p_grid_1'],
                'pGrid_2': results['p_grid_2'],
                'd2HGrid_1': results['d2h_grid_1'],
                'd2HGrid_2': results['d2h_grid_2'],
                'd18OGrid_1': results['d18o_grid_1'],
                'd18OGrid_2': results['d18o_grid_2'],
                'fractionPGrid': results['fraction_p_grid'],
                'isSampleSide01': results['is_sample_side_1'],
            })
        
        # Add sample data if present
        if len(input_data.get('sample_x', [])) > 0:
            save_data.update({
                'sampleLine': input_data.get('sample_line', np.array([])),
                'sampleLon': input_data['sample_lon'],
                'sampleLat': input_data['sample_lat'],
                'sampleX': input_data['sample_x'],
                'sampleY': input_data['sample_y'],
                'sampleD2H': input_data['sample_d2h'],
                'sampleD18O': input_data['sample_d18o'],
                'sampleDExcess': input_data.get('sample_d_excess', np.array([])),
                'sampleLC': input_data.get('sample_lc', np.array([])),
                'ijCatch': results['ij_catch'],
                'ptrCatch': results['ptr_catch'],
                'd2HPred': results.get('d2h_pred', []),
                'd18OPred': results.get('d18o_pred', []),
            })
        
        # Save to .mat file
        savemat(results_path, save_data)
        log(f"\nResults saved to: {results_path}")
        
        # Finalize log
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        log(f"\nComputation time: {duration:.2f} minutes")
        log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save log file
        save_log(log_filename, log_lines)
        
        return {
            'success': True,
            'results_path': results_path,
            'log_path': log_filename,
            'model_type': model_type,
            'chi_r2': results['chi_r2'],
            'results': results
        }
        
    except Exception as e:
        error_msg = f"\nERROR: {str(e)}"
        log(error_msg)
        import traceback
        log(traceback.format_exc())
        save_log(log_filename, log_lines)
        
        return {
            'success': False,
            'error': str(e),
            'log_path': log_filename
        }


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Run OPI (Orographic Precipitation Isotopes) simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m opi.run_opi_simulation runs/run001/run001.run
  python run_opi_simulation.py config/my_run.run --verbose
        """
    )
    
    parser.add_argument('run_file', help='Path to .run configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Print detailed progress (default: True)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress console output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet and args.verbose
    
    if not os.path.exists(args.run_file):
        print(f"Error: Run file not found: {args.run_file}")
        sys.exit(1)
    
    result = run_simulation(args.run_file, verbose=verbose)
    
    if result['success']:
        print(f"\nSimulation completed successfully!")
        print(f"Results: {result['results_path']}")
        print(f"Log: {result['log_path']}")
        sys.exit(0)
    else:
        print(f"\nSimulation failed: {result.get('error', 'Unknown error')}")
        print(f"See log for details: {result.get('log_path', 'N/A')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
