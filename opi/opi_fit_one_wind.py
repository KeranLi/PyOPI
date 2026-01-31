"""
One-Wind Parameter Fitting Function for OPI

This module implements the parameter fitting functionality for the 
one-wind model using CRS3 global optimization algorithm.
"""

import numpy as np
import os
from datetime import datetime

from .calc_one_wind import calc_one_wind
from .get_input import get_input
from .fmin_crs3 import fmin_crs3
from .constants import SD_RES_RATIO, HR


def opi_fit_one_wind(run_file_path=None, verbose=True, max_iterations=10000):
    """
    Performs parameter fitting for the one-wind OPI model.
    
    This function estimates the optimal parameters for the one-wind model
    using observational data and CRS3 global optimization.
    
    Parameters:
    -----------
    run_file_path : str, optional
        Path to the run file containing model parameters and file paths.
        If None, uses default parameters for demonstration.
    verbose : bool, optional
        Whether to print detailed progress information.
    max_iterations : int, optional
        Maximum number of optimization iterations.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'solution_params': fitted parameter values
        - 'derived_params': derived parameter values
        - 'misfit': final chi-square value
        - 'iterations': number of iterations performed
        - 'convergence': convergence status
        - 'message': result status message
    """
    if verbose:
        print("OPI One-Wind Parameter Fitting")
        print("=" * 35)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load run file or use defaults
    if run_file_path is None or not os.path.exists(run_file_path):
        if verbose:
            print("No run file specified, using default parameters for demonstration")
        run_data = get_default_run_data()
        run_dir = os.getcwd()
    else:
        run_dir = os.path.dirname(run_file_path)
        run_data = load_run_file(run_file_path)
        if verbose:
            print(f"Loaded run file: {run_file_path}")
    
    # Load input data
    data_path = os.path.join(run_dir, run_data['data_path'])
    
    if verbose:
        print(f"\nLoading input data...")
        print(f"Data path: {data_path}")
        print(f"Topography file: {run_data['topography_file']}")
    
    try:
        input_data = get_input(
            data_path=data_path,
            topo_file=run_data['topography_file'],
            r_tukey=run_data['r_tukey'],
            sample_file=run_data['sample_file'] if run_data['sample_file'].lower() != 'no' else None,
            sd_res_ratio=SD_RES_RATIO
        )
        use_real_data = True
    except FileNotFoundError:
        if verbose:
            print("Topography file not found. Creating synthetic data for demonstration.")
        input_data = create_synthetic_input()
        use_real_data = False
    
    if verbose:
        print(f"Grid size: {input_data['h_grid'].shape}")
        print(f"Number of samples: {len(input_data['sample_x'])}")
    
    # Prepare bounds and initial guess
    bounds_lower = np.array(run_data['param_constraints_min'])
    bounds_upper = np.array(run_data['param_constraints_max'])
    param_bounds = [(bounds_lower[i], bounds_upper[i]) for i in range(9)]
    
    # Initial guess (from run file or default)
    if run_data.get('initial_guess') is not None:
        initial_guess = np.array(run_data['initial_guess'])
    else:
        initial_guess = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
    
    # Determine which parameters are free
    if 'free_params' in run_data:
        free_params = np.array(run_data['free_params'], dtype=bool)
    else:
        free_params = np.ones(9, dtype=bool)  # All free by default
    
    n_free = np.sum(free_params)
    
    if verbose:
        print(f"\nParameter fitting:")
        print(f"  Total parameters: 9")
        print(f"  Free parameters: {n_free}")
        print(f"  Initial guess: {initial_guess}")
        print(f"  Optimization method: CRS3")
        print(f"  Max iterations: {max_iterations}")
    
    # Prepare sample data
    sample_d2h = input_data['sample_d2h']
    sample_d18o = input_data['sample_d18o']
    
    if len(sample_d2h) == 0:
        if verbose:
            print("No sample data available. Cannot perform fitting.")
        return {
            'solution_params': {},
            'derived_params': {},
            'misfit': np.nan,
            'iterations': 0,
            'convergence': False,
            'message': 'No sample data available'
        }
    
    # Calculate catchment nodes
    from .catchment_nodes import catchment_nodes
    ij_catch, ptr_catch = catchment_nodes(
        input_data['sample_x'], input_data['sample_y'],
        input_data['sample_lc'], input_data['x'], input_data['y'],
        input_data['h_grid']
    )
    
    # Prepare covariance matrix
    if input_data['cov'] is not None:
        cov = input_data['cov']
    else:
        cov = np.array([[1e-6, 0], [0, 1e-6]])
    
    # Objective function for optimization
    def objective(beta_free):
        """
        Calculate chi-square for given parameters.
        """
        # Reconstruct full parameter vector
        beta = initial_guess.copy()
        beta[free_params] = beta_free
        
        try:
            # Call the actual calculation function
            result = calc_one_wind(
                beta=beta,
                f_c=input_data['f_c'],
                h_r=input_data['h_r'],
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
                is_fit=True
            )
            
            # Extract chi-square
            chi_r2 = result[0]
            
            # Handle NaN or invalid values
            if np.isnan(chi_r2) or np.isinf(chi_r2):
                return 1e6
            
            return chi_r2
            
        except Exception as e:
            # Return large value if calculation fails
            return 1e6
    
    # Extract free parameter bounds
    free_bounds = [param_bounds[i] for i in range(9) if free_params[i]]
    free_initial = initial_guess[free_params]
    
    if verbose:
        print(f"\nStarting optimization...")
        print(f"Initial misfit: {objective(free_initial):.6f}")
    
    # Run CRS3 optimization
    try:
        opt_result = fmin_crs3(
            objective,
            free_bounds,
            mu=run_data.get('mu', 25),
            epsilon=run_data.get('epsilon', 1e-6),
            max_iter=max_iterations,
            random_state=42,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"  Final misfit: {opt_result.fun:.6f}")
            print(f"  Iterations: {opt_result.n_iter}")
            print(f"  Function evaluations: {opt_result.nfev}")
        
        # Reconstruct full solution vector
        solution_vector = initial_guess.copy()
        solution_vector[free_params] = opt_result.x
        
        # Calculate final results with full output
        final_result = calc_one_wind(
            beta=solution_vector,
            f_c=input_data['f_c'],
            h_r=input_data['h_r'],
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
        
        # Prepare output
        param_names = ['U', 'azimuth', 'T0', 'M', 'kappa', 'tau_c', 'd2h0', 'd_d2h0_d_lat', 'f_p0']
        solution_params = dict(zip(param_names, solution_vector))
        
        derived_params = {
            'chi_r2': float(opt_result.fun),
            'degrees_of_freedom': len(sample_d2h) - n_free,
            'T0_C': solution_vector[2] - 273.15,
            'function_evaluations': opt_result.nfev
        }
        
        return {
            'solution_params': solution_params,
            'derived_params': derived_params,
            'misfit': float(opt_result.fun),
            'iterations': opt_result.n_iter,
            'convergence': opt_result.success,
            'message': 'Optimization completed successfully',
            'solution_vector': solution_vector.tolist(),
            'final_results': {
                'precipitation': final_result[16] if len(final_result) > 16 else None,
                'd2h': final_result[22] if len(final_result) > 22 else None,
                'd18o': final_result[23] if len(final_result) > 23 else None
            }
        }
        
    except Exception as e:
        if verbose:
            print(f"Optimization failed: {e}")
        return {
            'solution_params': {},
            'derived_params': {},
            'misfit': np.nan,
            'iterations': 0,
            'convergence': False,
            'message': f'Optimization failed: {str(e)}'
        }


def get_default_run_data():
    """Get default run data for demonstration."""
    return {
        'run_title': 'Default Run',
        'data_path': 'data',
        'topography_file': 'topography.mat',
        'r_tukey': 0.0,
        'sample_file': 'no',
        'param_constraints_min': [0.1, -30, 265, 0, 0, 0, -15e-3, 0e-3, 1],
        'param_constraints_max': [25, 145, 295, 1.2, 1e6, 2500, 15e-3, 0e-3, 1],
        'initial_guess': [10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7],
        'mu': 25,
        'epsilon': 1e-6
    }


def load_run_file(run_file_path):
    """Load run file (simplified parser)."""
    with open(run_file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    
    return {
        'run_title': lines[0] if len(lines) > 0 else 'Run',
        'data_path': lines[2] if len(lines) > 2 else 'data',
        'topography_file': lines[4] if len(lines) > 4 else 'topography.mat',
        'r_tukey': float(lines[5]) if len(lines) > 5 else 0.0,
        'sample_file': lines[6] if len(lines) > 6 else 'no',
        'param_constraints_min': [0.1, -30, 265, 0, 0, 0, -15e-3, 0e-3, 1],
        'param_constraints_max': [25, 145, 295, 1.2, 1e6, 2500, 15e-3, 0e-3, 1],
        'initial_guess': [10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7],
        'mu': 25,
        'epsilon': 1e-6
    }


def create_synthetic_input():
    """Create synthetic input data for demonstration."""
    # Create grid
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create Gaussian mountain
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Create synthetic samples
    n_samples = 5
    sample_x = np.linspace(-30000, 30000, n_samples)
    sample_y = np.zeros(n_samples)
    
    # Sample elevations
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((y, x), h_grid, bounds_error=False, fill_value=0)
    sample_z = interp(np.column_stack([sample_y, sample_x]))
    
    # Synthetic isotope data (depletion with elevation)
    sample_d2h = (-50 - 0.05 * sample_z) * 1e-3  # permil to fraction
    sample_d18o = (-6 - 0.006 * sample_z) * 1e-3
    
    # Sample types
    sample_lc = np.array(['C'] * n_samples)
    
    # Origin
    lat0, lon0 = 45.0, 0.0
    
    # Coriolis
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    return {
        'lon': x / 111320,
        'lat': y / 111320,
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'lon0': lon0,
        'lat0': lat0,
        'sample_line': np.arange(1, n_samples + 1),
        'sample_lon': sample_x / 111320,
        'sample_lat': sample_y / 111320 + lat0,
        'sample_x': sample_x,
        'sample_y': sample_y,
        'sample_d2h': sample_d2h,
        'sample_d18o': sample_d18o,
        'sample_d_excess': sample_d2h - 8 * sample_d18o,
        'sample_lc': sample_lc,
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
        'sd_data_min': 1e-3,
        'sd_data_max': 28e-3,
        'cov': np.array([[1e-6, 0], [0, 1e-6]]),
        'f_c': f_c,
        'h_r': HR
    }


if __name__ == "__main__":
    result = opi_fit_one_wind(verbose=True, max_iterations=100)
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Misfit: {result['misfit']:.6f}")
    print(f"Converged: {result['convergence']}")
    print(f"Iterations: {result['iterations']}")
    print("\nSolution parameters:")
    for param, value in result['solution_params'].items():
        print(f"  {param}: {value}")
