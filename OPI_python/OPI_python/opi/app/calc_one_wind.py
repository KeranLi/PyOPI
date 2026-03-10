"""
Main function for calculating OPI results with one wind field
"""

import os
import numpy as np
from datetime import datetime
from scipy.io import loadmat, savemat
from ..calc_one_wind import calc_one_wind
from ..io.data_loader import get_input
from ..constants import TC2K, RADIUS_EARTH, M_PER_DEGREE, HR, SD_RES_RATIO
from ..io.coordinates import lonlat2xy, xy2lonlat
from ..catchment.nodes import catchment_nodes


def load_run_file(run_file_path):
    """
    Load parameters and file paths from a run file.
    """
    # Check if run file exists
    if not os.path.exists(run_file_path):
        print(f"Run file not found: {run_file_path}")
        return get_default_run_parameters()
    
    try:
        with open(run_file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Parse run file (similar to MATLAB format)
        params = {
            'run_title': lines[0] if len(lines) > 0 else 'Example Run',
            'parallel_mode': int(lines[1]) if len(lines) > 1 else 0,
            'data_path': lines[2] if len(lines) > 2 else 'data',
            'aux_path': lines[3] if len(lines) > 3 else '',
            'topography_file': lines[4] if len(lines) > 4 else 'topography.mat',
            'r_tukey': float(lines[5]) if len(lines) > 5 else 0.25,
            'sample_file': lines[6] if len(lines) > 6 else 'no',
            'divide_file': lines[7] if len(lines) > 7 else 'no',
        }
        
        # Parse map limits (line 9)
        if len(lines) > 8:
            map_limits = lines[8].strip('[]').split()
            params['map_limits'] = [float(x) for x in map_limits]
        else:
            params['map_limits'] = [0, 0, 0, 0]
        
        # Path origin (line 10)
        params['path_origin'] = lines[9] if len(lines) > 9 else 'map'
        
        # Search parameters (line 11)
        if len(lines) > 10:
            search_params = lines[10].strip('[]').split()
            params['search_params'] = [float(x) for x in search_params]
        else:
            params['search_params'] = [10, 1e-4]
        
        # Restart file (line 12)
        params['restart_file'] = lines[11] if len(lines) > 11 else 'no'
        
        # Parameter labels, exponents, constraints (lines 13-16)
        if len(lines) > 12:
            params['param_labels'] = lines[12].split('|')
        else:
            params['param_labels'] = ['U', 'azimuth', 'T0', 'M', 'kappa', 'tau_c', 'd2H0', 'dd2H0/dlat', 'fP']
        
        if len(lines) > 13:
            exponents = lines[13].strip('[]').split()
            params['param_exponents'] = [int(x) for x in exponents]
        else:
            params['param_exponents'] = [0, 0, 0, 0, 6, 0, -3, -3, 0]
        
        # Constraints
        if len(lines) > 14:
            lower_bounds = lines[14].strip('[]').split()
            params['param_constraints_min'] = [float(x) for x in lower_bounds]
        else:
            params['param_constraints_min'] = [0.1, -30, 265, 0, 0, 0, -15e-3, 0e-3, 1]
        
        if len(lines) > 15:
            upper_bounds = lines[15].strip('[]').split()
            params['param_constraints_max'] = [float(x) for x in upper_bounds]
        else:
            params['param_constraints_max'] = [25, 145, 295, 1.2, 1e6, 2500, 15e-3, 0e-3, 1]
        
        # Solution vector (if present)
        if len(lines) > 16:
            solution = lines[16].strip('[]').split()
            params['solution'] = [float(x) for x in solution]
        else:
            params['solution'] = None
        
        return params
    except Exception as e:
        print(f"Error parsing run file: {e}")
        return get_default_run_parameters()


def get_default_run_parameters():
    """Return default parameters for demonstration purposes."""
    return {
        'run_title': 'Example Run',
        'parallel_mode': 0,
        'data_path': 'data',
        'aux_path': '',
        'topography_file': 'topography.mat',
        'r_tukey': 0.0,
        'sample_file': 'no',
        'divide_file': 'no',
        'map_limits': [0, 0, 0, 0],
        'path_origin': 'map',
        'search_params': [10, 1e-4],
        'restart_file': 'no',
        'param_labels': ['U (m/s)', 'azimuth', 'T_0 (K)', 'M', 'kappa (km^2/s)', 
                         'tau_c (s)', 'd2H0', 'dd2H0/dlat', 'fP'],
        'param_exponents': [0, 0, 0, 0, 6, 0, -3, -3, 0],
        'param_constraints_min': [0.1, -30, 265, 0, 0, 0, -15e-3, 0e-3, 1],
        'param_constraints_max': [25, 145, 295, 1.2, 1e6, 2500, 15e-3, 0e-3, 1],
        'solution': None
    }


def opi_calc_one_wind(run_file_path=None, solution_vector=None, verbose=True):
    """
    Main function for calculating OPI results with one wind field
    
    Parameters:
    -----------
    run_file_path : str, optional
        Path to the run file containing model parameters
    solution_vector : array-like, optional
        9-parameter solution vector [U, azimuth, T0, M, kappa, tau_c, d2h0, d_d2h0_d_lat, f_p0]
    verbose : bool, optional
        Whether to print detailed progress information
    
    Returns:
    --------
    dict
        Dictionary containing solution and derived parameters
    """
    if verbose:
        print("OPI Calculation for One Wind Field")
        print("="*35)
    
    # Load run file or use defaults
    if run_file_path and os.path.exists(run_file_path):
        run_data = load_run_file(run_file_path)
        run_dir = os.path.dirname(run_file_path)
        if verbose:
            print(f"Loading run file: {run_file_path}")
    else:
        run_data = get_default_run_parameters()
        run_dir = os.getcwd()
        if verbose:
            print("No run file specified, using default parameters")
    
    if verbose:
        print(f"Run title: {run_data['run_title']}")
    
    # Get data path
    data_path = os.path.join(run_dir, run_data['data_path'])
    
    # Load input data (topography and samples)
    if verbose:
        print("\nLoading input data...")
    
    try:
        input_data = get_input(
            data_path=data_path,
            topo_file=run_data['topography_file'],
            r_tukey=run_data['r_tukey'],
            sample_file=run_data['sample_file'] if run_data['sample_file'].lower() != 'no' else None,
            sd_res_ratio=SD_RES_RATIO
        )
    except FileNotFoundError:
        print(f"Topography file not found. Creating synthetic data for demonstration.")
        input_data = create_synthetic_data()
    
    if verbose:
        print(f"Grid size: {input_data['h_grid'].shape}")
        print(f"Maximum elevation: {input_data['h_grid'].max():.0f} m")
        print(f"Origin (lon0, lat0): ({input_data['lon0']:.4f}, {input_data['lat0']:.4f})")
        print(f"Coriolis parameter: {input_data['f_c']*1e4:.4f} x 10^-4 rad/s")
    
    # Get solution vector
    if solution_vector is not None:
        beta = np.array(solution_vector)
    elif run_data.get('solution') is not None:
        beta = np.array(run_data['solution'])
    else:
        # Default parameters
        beta = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5.0e-3, -2.0e-3, 0.7])
    
    if verbose:
        print(f"\nSolution parameters:")
        param_names = ['U (m/s)', 'Azimuth (deg)', 'T0 (K)', 'M', 'kappa (m^2/s)', 
                       'tau_c (s)', 'd2H0', 'dd2H0/dlat', 'fP']
        for name, val in zip(param_names, beta):
            print(f"  {name}: {val}")
    
    # Prepare sample data for calc_one_wind
    sample_d2h = input_data['sample_d2h']
    sample_d18o = input_data['sample_d18o']
    
    # If no samples, create empty arrays
    if len(sample_d2h) == 0:
        sample_d2h = np.array([np.nan])
        sample_d18o = np.array([np.nan])
    
    # Calculate catchment nodes if samples exist
    if len(input_data['sample_x']) > 0:
        ij_catch, ptr_catch = catchment_nodes(
            input_data['sample_x'], input_data['sample_y'], 
            input_data['sample_lc'], input_data['x'], input_data['y'], 
            input_data['h_grid']
        )
    else:
        ij_catch = []
        ptr_catch = [0]
    
    # Number of free parameters
    n_parameters_free = 9
    is_fit = False
    
    # Prepare covariance matrix
    if input_data['cov'] is not None:
        cov = input_data['cov']
    else:
        # Default covariance
        cov = np.array([[1e-6, 0], [0, 1e-6]])
    
    # Call the actual calculation function
    if verbose:
        print("\nRunning OPI calculation...")
    
    try:
        result = calc_one_wind(
            beta=beta,
            f_c=input_data['f_c'],
            h_r=input_data['h_r'],
            x=input_data['x'],
            y=input_data['y'],
            lat=np.array([input_data['lat0']]),  # Use origin as reference
            lat0=input_data['lat0'],
            h_grid=input_data['h_grid'],
            b_mwl_sample=input_data['b_mwl_sample'],
            ij_catch=ij_catch,
            ptr_catch=ptr_catch,
            sample_d2h=sample_d2h,
            sample_d18o=sample_d18o,
            cov=cov,
            n_parameters_free=n_parameters_free,
            is_fit=is_fit
        )
        
        # Unpack results
        (chi_r2, nu, std_residuals, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
         rho_s0, h_s, rho0, h_rho, d18o0, d_d18o0_d_lat, tau_f, p_grid, f_m_grid, r_h_grid,
         evap_d2h_grid, u_evap_d2h_grid, evap_d18o_grid, u_evap_d18o_grid,
         d2h_grid, d18o_grid, i_wet, d2h_pred, d18o_pred) = result
        
        if verbose:
            print("\nCalculation completed successfully!")
            chi_r2_display = chi_r2 if (isinstance(chi_r2, float) and not np.isnan(chi_r2)) else 'N/A'
            print(f"Chi-square: {chi_r2_display}")
            print(f"Mean precipitation: {p_grid.mean():.6f} kg/m^2/s")
            print(f"d2H range: {d2h_grid.min()*1000:.1f} to {d2h_grid.max()*1000:.1f} permil")
            print(f"d18O range: {d18o_grid.min()*1000:.1f} to {d18o_grid.max()*1000:.1f} permil")
        
        # Create solution parameters dict
        param_names = ['U (m/s)', 'Azimuth (deg)', 'T0 (K)', 'M', 'kappa (m^2/s)', 
                       'tau_c (s)', 'd2H0', 'dd2H0/dlat', 'fP']
        solution_params_dict = dict(zip(param_names, beta))
        
        # Calculate derived parameters
        derived_params = {
            'precip_fraction': float(np.mean(f_m_grid)),
            'mean_isotope_d2h': float(np.mean(d2h_grid)) * 1000,
            'mean_isotope_d18o': float(np.mean(d18o_grid)) * 1000,
            'tau_f': float(tau_f),
            'h_s': float(h_s),
            'rho_s0': float(rho_s0)
        }
        
        return {
            'solution_params': solution_params_dict,
            'derived_params': derived_params,
            'results': {
                'precipitation': p_grid,
                'moisture_ratio': f_m_grid,
                'd2h': d2h_grid,
                'd18o': d18o_grid,
                'x': input_data['x'],
                'y': input_data['y'],
                'h_grid': input_data['h_grid']
            },
            'chi_r2': chi_r2,
            'nu': nu
        }
    
    except Exception as e:
        if verbose:
            print(f"\nError during calculation: {e}")
            import traceback
            traceback.print_exc()
        
        # Return minimal result
        param_names = ['U (m/s)', 'Azimuth (deg)', 'T0 (K)', 'M', 'kappa (m^2/s)', 
                       'tau_c (s)', 'd2H0', 'dd2H0/dlat', 'fP']
        solution_params_dict = dict(zip(param_names, beta))
        
        return {
            'solution_params': solution_params_dict,
            'derived_params': {},
            'results': {},
            'chi_r2': np.nan,
            'nu': 0
        }


def create_synthetic_data():
    """Create synthetic data for demonstration."""
    # Create grid
    nx, ny = 100, 100
    x = np.linspace(-50000, 50000, nx)
    y = np.linspace(-50000, 50000, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create Gaussian mountain
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Create longitude/latitude (approximate)
    lon = x / M_PER_DEGREE
    lat = y / M_PER_DEGREE
    
    lon0 = 0.0
    lat0 = 45.0
    
    # Coriolis parameter
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    return {
        'lon': lon,
        'lat': lat,
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'lon0': lon0,
        'lat0': lat0,
        'sample_line': np.array([]),
        'sample_lon': np.array([]),
        'sample_lat': np.array([]),
        'sample_x': np.array([]),
        'sample_y': np.array([]),
        'sample_d2h': np.array([]),
        'sample_d18o': np.array([]),
        'sample_d_excess': np.array([]),
        'sample_lc': np.array([]),
        'sample_line_alt': np.array([]),
        'sample_lon_alt': np.array([]),
        'sample_lat_alt': np.array([]),
        'sample_x_alt': np.array([]),
        'sample_y_alt': np.array([]),
        'sample_d2h_alt': np.array([]),
        'sample_d18o_alt': np.array([]),
        'sample_d_excess_alt': np.array([]),
        'sample_lc_alt': np.array([]),
        'b_mwl_sample': np.array([9.47e-3, 8.03]),
        'sd_data_min': None,
        'sd_data_max': None,
        'cov': None,
        'f_c': f_c,
        'h_r': HR
    }


def main():
    """Main entry point."""
    return opi_calc_one_wind(verbose=True)


if __name__ == "__main__":
    main()
