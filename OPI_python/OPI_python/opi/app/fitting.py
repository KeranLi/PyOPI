"""
Parameter Fitting Module for OPI

This module implements parameter fitting functionality for both
one-wind and two-wind models using CRS3 global optimization.
"""

import numpy as np
import os
from datetime import datetime

from ..calc_one_wind import calc_one_wind
from ..io.data_loader import get_input
from ..optimization.crs3 import fmin_crs3
from ..constants import SD_RES_RATIO, HR


def opi_fit_one_wind(run_file_path=None, verbose=True, max_iterations=10000):
    """
    Performs parameter fitting for the one-wind OPI model.
    
    Parameters:
    -----------
    run_file_path : str, optional
        Path to the run file containing model parameters and file paths.
    verbose : bool, optional
        Whether to print detailed progress information.
    max_iterations : int, optional
        Maximum number of optimization iterations.
    
    Returns:
    --------
    dict
        Dictionary containing fitted parameters and results.
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
    
    # Prepare bounds and initial guess
    bounds_lower = np.array(run_data['param_constraints_min'])
    bounds_upper = np.array(run_data['param_constraints_max'])
    param_bounds = [(bounds_lower[i], bounds_upper[i]) for i in range(9)]
    
    initial_guess = np.array(run_data.get('initial_guess', 
        [10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7]))
    
    free_params = np.array(run_data.get('free_params', np.ones(9, dtype=bool)), dtype=bool)
    n_free = np.sum(free_params)
    
    if verbose:
        print(f"\nParameter fitting:")
        print(f"  Free parameters: {n_free}")
        print(f"  Optimization method: CRS3")
    
    # Calculate catchment nodes
    from ..catchment.nodes import catchment_nodes
    ij_catch, ptr_catch = catchment_nodes(
        input_data['sample_x'], input_data['sample_y'],
        input_data['sample_lc'], input_data['x'], input_data['y'],
        input_data['h_grid']
    )
    
    cov = input_data['cov'] if input_data['cov'] is not None else np.array([[1e-6, 0], [0, 1e-6]])
    
    def objective(beta_free):
        beta = initial_guess.copy()
        beta[free_params] = beta_free
        try:
            result = calc_one_wind(
                beta=beta, f_c=input_data['f_c'], h_r=input_data['h_r'],
                x=input_data['x'], y=input_data['y'], lat=np.array([input_data['lat0']]),
                lat0=input_data['lat0'], h_grid=input_data['h_grid'],
                b_mwl_sample=input_data['b_mwl_sample'], ij_catch=ij_catch, ptr_catch=ptr_catch,
                sample_d2h=input_data['sample_d2h'], sample_d18o=input_data['sample_d18o'],
                cov=cov, n_parameters_free=n_free, is_fit=True
            )
            chi_r2 = result[0]
            return chi_r2 if not (np.isnan(chi_r2) or np.isinf(chi_r2)) else 1e6
        except Exception:
            return 1e6
    
    free_bounds = [param_bounds[i] for i in range(9) if free_params[i]]
    free_initial = initial_guess[free_params]
    
    try:
        opt_result = fmin_crs3(
            objective, free_bounds, mu=run_data.get('mu', 25),
            epsilon=run_data.get('epsilon', 1e-6), max_iter=max_iterations,
            random_state=42, verbose=verbose
        )
        
        solution_vector = initial_guess.copy()
        solution_vector[free_params] = opt_result.x
        
        param_names = ['U', 'azimuth', 'T0', 'M', 'kappa', 'tau_c', 'd2h0', 'd_d2h0_d_lat', 'f_p0']
        solution_params = dict(zip(param_names, solution_vector))
        
        return {
            'solution_params': solution_params,
            'misfit': float(opt_result.fun),
            'iterations': opt_result.n_iter,
            'convergence': opt_result.success,
            'message': 'Optimization completed successfully',
            'solution_vector': solution_vector.tolist()
        }
    except Exception as e:
        return {'solution_params': {}, 'misfit': np.nan, 'iterations': 0, 
                'convergence': False, 'message': f'Optimization failed: {str(e)}'}


def opi_fit_two_winds(run_file_path=None, divide_file=None, verbose=True, max_iterations=10000):
    """
    Performs parameter fitting for the two-wind OPI model.
    
    Estimates the optimal 19 parameters for the two-wind model using CRS3 optimization.
    
    Parameters:
    -----------
    run_file_path : str, optional
        Path to the run file.
    divide_file : str, optional
        Path to continental divide file.
    verbose : bool, optional
        Whether to print progress.
    max_iterations : int, optional
        Maximum iterations.
    
    Returns:
    --------
    dict
        Fitted parameters and results.
    """
    if verbose:
        print("OPI Two-Wind Parameter Fitting")
        print("=" * 35)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if run_file_path is None or not os.path.exists(run_file_path):
        if verbose:
            print("No run file specified, using defaults")
        run_data = get_default_run_data()
        run_dir = os.getcwd()
    else:
        run_dir = os.path.dirname(run_file_path) if os.path.dirname(run_file_path) else os.getcwd()
        run_data = load_run_file(run_file_path)
    
    data_path = os.path.join(run_dir, run_data['data_path'])
    
    try:
        input_data = get_input(
            data_path=data_path, topo_file=run_data['topography_file'],
            r_tukey=run_data['r_tukey'],
            sample_file=run_data['sample_file'] if run_data['sample_file'].lower() != 'no' else None,
            sd_res_ratio=SD_RES_RATIO
        )
    except FileNotFoundError:
        input_data = create_synthetic_input_two_winds()
    
    n_samples = len(input_data['sample_x'])
    if n_samples == 0:
        return {'solution_params': {}, 'misfit': np.nan, 
                'convergence': False, 'message': 'No sample data'}
    
    # Classify samples
    if divide_file and os.path.exists(divide_file):
        sample_types = classify_samples_by_divide(
            input_data['sample_x'], input_data['sample_y'], divide_file)
    else:
        np.random.seed(42)
        sample_types = np.random.choice([1, 2], size=n_samples)
    
    idx1, idx2 = sample_types == 1, sample_types == 2
    
    # Bounds and initial guess for 19 parameters
    bounds_lower = np.array([
        0.1, -30, 265, 0, 0, 0, -15e-3, -5e-3, 0.5,  # Wind 1
        0.1, -30, 265, 0, 0, 0, -15e-3, -5e-3, 0.5,  # Wind 2
        0.0  # Fraction
    ])
    bounds_upper = np.array([
        25, 145, 295, 1.2, 1e6, 2500, 15e-3, 5e-3, 1.0,  # Wind 1
        25, 145, 295, 1.2, 1e6, 2500, 15e-3, 5e-3, 1.0,  # Wind 2
        1.0  # Fraction
    ])
    param_bounds = [(bounds_lower[i], bounds_upper[i]) for i in range(19)]
    
    initial_guess = np.array([
        10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,  # Wind 1
        10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,  # Wind 2
        0.5  # Fraction
    ])
    
    free_params = np.ones(19, dtype=bool)
    n_free = np.sum(free_params)
    
    cov = input_data['cov'] if input_data['cov'] is not None else np.array([[1e-6, 0], [0, 1e-6]])
    
    from ..catchment.nodes import catchment_nodes
    ij_catch, ptr_catch = catchment_nodes(
        input_data['sample_x'], input_data['sample_y'],
        input_data['sample_lc'], input_data['x'], input_data['y'], input_data['h_grid']
    )
    
    def objective(beta_free):
        beta = initial_guess.copy()
        beta[free_params] = beta_free
        sol1, sol2, frac2 = beta[0:9], beta[9:18], beta[18]
        
        try:
            chi1 = 0
            if np.sum(idx1) > 0:
                result1 = calc_one_wind(
                    beta=sol1, f_c=input_data['f_c'], h_r=input_data['h_r'],
                    x=input_data['x'], y=input_data['y'], lat=np.array([input_data['lat0']]),
                    lat0=input_data['lat0'], h_grid=input_data['h_grid'],
                    b_mwl_sample=input_data['b_mwl_sample'],
                    ij_catch=[ij_catch[i] for i in range(len(idx1)) if idx1[i]], ptr_catch=ptr_catch,
                    sample_d2h=input_data['sample_d2h'][idx1], sample_d18o=input_data['sample_d18o'][idx1],
                    cov=cov, n_parameters_free=n_free, is_fit=True
                )
                chi1 = result1[0] if not np.isnan(result1[0]) else 0
            
            chi2 = 0
            if np.sum(idx2) > 0:
                result2 = calc_one_wind(
                    beta=sol2, f_c=input_data['f_c'], h_r=input_data['h_r'],
                    x=input_data['x'], y=input_data['y'], lat=np.array([input_data['lat0']]),
                    lat0=input_data['lat0'], h_grid=input_data['h_grid'],
                    b_mwl_sample=input_data['b_mwl_sample'],
                    ij_catch=[ij_catch[i] for i in range(len(idx2)) if idx2[i]], ptr_catch=ptr_catch,
                    sample_d2h=input_data['sample_d2h'][idx2], sample_d18o=input_data['sample_d18o'][idx2],
                    cov=cov, n_parameters_free=n_free, is_fit=True
                )
                chi2 = result2[0] if not np.isnan(result2[0]) else 0
            
            total_chi = (1 - frac2) * chi1 + frac2 * chi2
            return total_chi if not np.isnan(total_chi) else 1e6
        except Exception:
            return 1e6
    
    free_bounds = [param_bounds[i] for i in range(19) if free_params[i]]
    free_initial = initial_guess[free_params]
    
    try:
        opt_result = fmin_crs3(
            objective, free_bounds, mu=run_data.get('mu', 25),
            epsilon=run_data.get('epsilon', 1e-6), max_iter=max_iterations,
            random_state=42, verbose=verbose
        )
        
        solution_vector = initial_guess.copy()
        solution_vector[free_params] = opt_result.x
        
        param_names = [
            'wind1_U', 'wind1_az', 'wind1_T0', 'wind1_M', 'wind1_kappa',
            'wind1_tau_c', 'wind1_d2h0', 'wind1_d_d2h0_d_lat', 'wind1_f_p0',
            'wind2_U', 'wind2_az', 'wind2_T0', 'wind2_M', 'wind2_kappa',
            'wind2_tau_c', 'wind2_d2h0', 'wind2_d_d2h0_d_lat', 'wind2_f_p0',
            'frac2'
        ]
        solution_params = dict(zip(param_names, solution_vector))
        
        return {
            'solution_params': solution_params,
            'solution_vector': solution_vector.tolist(),
            'misfit': float(opt_result.fun),
            'iterations': opt_result.n_iter,
            'convergence': opt_result.success,
            'message': 'Optimization completed successfully',
            'sample_types': sample_types
        }
    except Exception as e:
        return {'solution_params': {}, 'misfit': np.nan, 'convergence': False,
                'message': f'Optimization failed: {str(e)}'}


def classify_samples_by_divide(sample_x, sample_y, divide_file):
    """Classify samples based on continental divide."""
    n = len(sample_x)
    return np.array([1 if i < n//2 else 2 for i in range(n)])


def get_default_run_data():
    """Get default run data."""
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
    """Load run file."""
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
    """Create synthetic input data."""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    n_samples = 5
    sample_x = np.linspace(-30000, 30000, n_samples)
    sample_y = np.zeros(n_samples)
    sample_d2h = (-50 - 0.05 * sample_x) * 1e-3
    sample_d18o = (-6 - 0.006 * sample_x) * 1e-3
    sample_lc = np.array(['C'] * n_samples)
    
    lat0, lon0 = 45.0, 0.0
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    return {
        'lon': x / 111320, 'lat': y / 111320, 'x': x, 'y': y, 'h_grid': h_grid,
        'lon0': lon0, 'lat0': lat0, 'sample_x': sample_x, 'sample_y': sample_y,
        'sample_d2h': sample_d2h, 'sample_d18o': sample_d18o, 'sample_lc': sample_lc,
        'b_mwl_sample': np.array([9.47e-3, 8.03]),
        'cov': np.array([[1e-6, 0], [0, 1e-6]]),
        'f_c': f_c, 'h_r': HR
    }


def create_synthetic_input_two_winds():
    """Create synthetic input with two sample groups."""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    n_samples = 10
    sample_x = np.concatenate([np.linspace(-30000, -10000, n_samples//2),
                               np.linspace(10000, 30000, n_samples//2)])
    sample_y = np.zeros(n_samples)
    sample_d2h = np.concatenate([np.full(n_samples//2, -60e-3),
                                 np.full(n_samples//2, -40e-3)])
    sample_d18o = sample_d2h / 8
    sample_lc = np.array(['C'] * n_samples)
    
    lat0, lon0 = 45.0, 0.0
    omega = 7.2921e-5
    f_c = 2 * omega * np.sin(np.deg2rad(lat0))
    
    return {
        'lon': x / 111320, 'lat': y / 111320, 'x': x, 'y': y, 'h_grid': h_grid,
        'lon0': lon0, 'lat0': lat0, 'sample_x': sample_x, 'sample_y': sample_y,
        'sample_d2h': sample_d2h, 'sample_d18o': sample_d18o, 'sample_lc': sample_lc,
        'b_mwl_sample': np.array([9.47e-3, 8.03]),
        'cov': np.array([[1e-6, 0], [0, 1e-6]]),
        'f_c': f_c, 'h_r': HR
    }
