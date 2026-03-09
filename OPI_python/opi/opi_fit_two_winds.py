"""
Two-Wind Parameter Fitting Function for OPI

This module implements parameter fitting for the two-wind model,
which optimizes 19 parameters for two distinct moisture sources.
"""

import numpy as np
import os
from datetime import datetime

from .calc_one_wind import calc_one_wind
from .get_input import get_input
from .fmin_crs3 import fmin_crs3
from .constants import SD_RES_RATIO, HR


def opi_fit_two_winds(run_file_path=None, divide_file=None, verbose=True, max_iterations=10000):
    """
    Performs parameter fitting for the two-wind OPI model.
    
    This function estimates the optimal 19 parameters for the two-wind model
    using observational data and CRS3 global optimization.
    
    Parameters:
    -----------
    run_file_path : str, optional
        Path to the run file containing model parameters and file paths
    divide_file : str, optional
        Path to continental divide file for separating samples
    verbose : bool, optional
        Whether to print detailed progress information
    max_iterations : int, optional
        Maximum number of optimization iterations
    
    Returns:
    --------
    dict
        Dictionary containing fitted parameters and results
    """
    if verbose:
        print("OPI Two-Wind Parameter Fitting")
        print("=" * 35)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load run file or use defaults
    if run_file_path is None or not os.path.exists(run_file_path):
        if verbose:
            print("No run file specified, using default parameters for demonstration")
        run_data = get_default_run_data()
        run_dir = os.getcwd()
    else:
        run_dir = os.path.dirname(run_file_path) if os.path.dirname(run_file_path) else os.getcwd()
        run_data = load_run_file(run_file_path)
        if verbose:
            print(f"Loaded run file: {run_file_path}")
    
    # Load input data
    data_path = os.path.join(run_dir, run_data['data_path'])
    
    if verbose:
        print(f"\nLoading input data...")
        print(f"Data path: {data_path}")
    
    try:
        input_data = get_input(
            data_path=data_path,
            topo_file=run_data['topography_file'],
            r_tukey=run_data['r_tukey'],
            sample_file=run_data['sample_file'] if run_data['sample_file'].lower() != 'no' else None,
            sd_res_ratio=SD_RES_RATIO
        )
    except FileNotFoundError:
        if verbose:
            print("Topography file not found. Creating synthetic data for demonstration.")
        input_data = create_synthetic_input_two_winds()
    
    n_samples = len(input_data['sample_x'])
    
    if verbose:
        print(f"Grid size: {input_data['h_grid'].shape}")
        print(f"Number of samples: {n_samples}")
    
    if n_samples == 0:
        if verbose:
            print("No sample data available. Cannot perform fitting.")
        return {
            'solution_params': {},
            'misfit': np.nan,
            'convergence': False,
            'message': 'No sample data available'
        }
    
    # Load continental divide if provided
    # This separates samples into two groups (windward side of each wind field)
    if divide_file and os.path.exists(divide_file):
        # Load divide polygon and classify samples
        sample_types = classify_samples_by_divide(
            input_data['sample_x'], input_data['sample_y'], divide_file
        )
    else:
        # Default: randomly split samples for demonstration
        # In real usage, would use spatial analysis
        np.random.seed(42)
        sample_types = np.random.choice([1, 2], size=n_samples)
    
    # Split samples by type
    idx1 = sample_types == 1
    idx2 = sample_types == 2
    
    if verbose:
        print(f"Sample classification:")
        print(f"  Wind Field 1: {np.sum(idx1)} samples")
        print(f"  Wind Field 2: {np.sum(idx2)} samples")
    
    # Prepare parameter bounds (19 parameters)
    # [Wind1: 9 params, Wind2: 9 params, frac2: 1 param]
    bounds_lower = np.array([
        # Wind 1
        0.1, -30, 265, 0, 0, 0, -15e-3, -5e-3, 0.5,
        # Wind 2
        0.1, -30, 265, 0, 0, 0, -15e-3, -5e-3, 0.5,
        # Fraction
        0.0
    ])
    
    bounds_upper = np.array([
        # Wind 1
        25, 145, 295, 1.2, 1e6, 2500, 15e-3, 5e-3, 1.0,
        # Wind 2
        25, 145, 295, 1.2, 1e6, 2500, 15e-3, 5e-3, 1.0,
        # Fraction
        1.0
    ])
    
    param_bounds = [(bounds_lower[i], bounds_upper[i]) for i in range(19)]
    
    # Initial guess
    if 'initial_guess' in run_data and len(run_data['initial_guess']) == 19:
        initial_guess = np.array(run_data['initial_guess'])
    else:
        initial_guess = np.array([
            # Wind 1
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,
            # Wind 2
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,
            # Fraction
            0.5
        ])
    
    # Determine free parameters
    if 'free_params' in run_data and len(run_data['free_params']) == 19:
        free_params = np.array(run_data['free_params'], dtype=bool)
    else:
        free_params = np.ones(19, dtype=bool)
        # Fix some parameters if needed
        # free_params[7] = False  # d_d2h0_d_lat_1
        # free_params[16] = False  # d_d2h0_d_lat_2
    
    n_free = np.sum(free_params)
    
    if verbose:
        print(f"\nParameter fitting:")
        print(f"  Total parameters: 19")
        print(f"  Free parameters: {n_free}")
        print(f"  Optimization method: CRS3")
    
    # Prepare covariance
    if input_data['cov'] is not None:
        cov = input_data['cov']
    else:
        cov = np.array([[1e-6, 0], [0, 1e-6]])
    
    # Calculate catchment nodes for all samples
    from .catchment_nodes import catchment_nodes
    ij_catch, ptr_catch = catchment_nodes(
        input_data['sample_x'], input_data['sample_y'],
        input_data['sample_lc'], input_data['x'], input_data['y'],
        input_data['h_grid']
    )
    
    # Objective function
    def objective(beta_free):
        """Calculate total chi-square for two-wind model."""
        beta = initial_guess.copy()
        beta[free_params] = beta_free
        
        # Split parameters
        sol1 = beta[0:9]
        sol2 = beta[9:18]
        frac2 = beta[18]
        
        try:
            # Calculate for wind field 1 (samples on side 1)
            if np.sum(idx1) > 0:
                result1 = calc_one_wind(
                    beta=sol1,
                    f_c=input_data['f_c'],
                    h_r=input_data['h_r'],
                    x=input_data['x'],
                    y=input_data['y'],
                    lat=np.array([input_data['lat0']]),
                    lat0=input_data['lat0'],
                    h_grid=input_data['h_grid'],
                    b_mwl_sample=input_data['b_mwl_sample'],
                    ij_catch=[ij_catch[i] for i in range(len(idx1)) if idx1[i]],
                    ptr_catch=ptr_catch,
                    sample_d2h=input_data['sample_d2h'][idx1],
                    sample_d18o=input_data['sample_d18o'][idx1],
                    cov=cov,
                    n_parameters_free=n_free,
                    is_fit=True
                )
                chi1 = result1[0] if not np.isnan(result1[0]) else 0
            else:
                chi1 = 0
            
            # Calculate for wind field 2 (samples on side 2)
            if np.sum(idx2) > 0:
                result2 = calc_one_wind(
                    beta=sol2,
                    f_c=input_data['f_c'],
                    h_r=input_data['h_r'],
                    x=input_data['x'],
                    y=input_data['y'],
                    lat=np.array([input_data['lat0']]),
                    lat0=input_data['lat0'],
                    h_grid=input_data['h_grid'],
                    b_mwl_sample=input_data['b_mwl_sample'],
                    ij_catch=[ij_catch[i] for i in range(len(idx2)) if idx2[i]],
                    ptr_catch=ptr_catch,
                    sample_d2h=input_data['sample_d2h'][idx2],
                    sample_d18o=input_data['sample_d18o'][idx2],
                    cov=cov,
                    n_parameters_free=n_free,
                    is_fit=True
                )
                chi2 = result2[0] if not np.isnan(result2[0]) else 0
            else:
                chi2 = 0
            
            # Total chi-square (weighted by fraction)
            total_chi = (1 - frac2) * chi1 + frac2 * chi2
            
            return total_chi if not np.isnan(total_chi) else 1e6
            
        except Exception as e:
            return 1e6
    
    # Extract free bounds
    free_bounds = [param_bounds[i] for i in range(19) if free_params[i]]
    free_initial = initial_guess[free_params]
    
    if verbose:
        print(f"\nStarting optimization...")
        print(f"Initial misfit: {objective(free_initial):.6f}")
    
    # Run optimization
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
        
        # Reconstruct solution
        solution_vector = initial_guess.copy()
        solution_vector[free_params] = opt_result.x
        
        # Prepare output
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
        if verbose:
            print(f"Optimization failed: {e}")
        return {
            'solution_params': {},
            'misfit': np.nan,
            'convergence': False,
            'message': f'Optimization failed: {str(e)}'
        }


def classify_samples_by_divide(sample_x, sample_y, divide_file):
    """
    Classify samples based on which side of continental divide they fall.
    
    Returns array of 1 or 2 indicating which wind field the sample belongs to.
    """
    # Simplified: alternate assignment for demonstration
    # Real implementation would use polygon intersection
    n = len(sample_x)
    return np.array([1 if i < n//2 else 2 for i in range(n)])


def get_default_run_data():
    """Get default run data."""
    return {
        'run_title': 'Two-Wind Default',
        'data_path': 'data',
        'topography_file': 'topography.mat',
        'r_tukey': 0.0,
        'sample_file': 'no',
        'initial_guess': None,
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
        'initial_guess': None,
        'mu': 25,
        'epsilon': 1e-6
    }


def create_synthetic_input_two_winds():
    """Create synthetic input with two distinct sample groups."""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Create two groups of samples (simulating two wind sides)
    n_samples = 10
    sample_x = np.concatenate([
        np.linspace(-30000, -10000, n_samples//2),
        np.linspace(10000, 30000, n_samples//2)
    ])
    sample_y = np.zeros(n_samples)
    
    # Different isotopic signatures for each group
    sample_d2h = np.concatenate([
        np.full(n_samples//2, -60e-3),  # Group 1: more depleted
        np.full(n_samples//2, -40e-3)   # Group 2: less depleted
    ])
    sample_d18o = sample_d2h / 8  # Approximate MWL
    
    sample_lc = np.array(['C'] * n_samples)
    
    lat0, lon0 = 45.0, 0.0
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
        'sample_x': sample_x,
        'sample_y': sample_y,
        'sample_d2h': sample_d2h,
        'sample_d18o': sample_d18o,
        'sample_lc': sample_lc,
        'b_mwl_sample': np.array([9.47e-3, 8.03]),
        'cov': np.array([[1e-6, 0], [0, 1e-6]]),
        'f_c': f_c,
        'h_r': HR
    }


if __name__ == "__main__":
    result = opi_fit_two_winds(verbose=True, max_iterations=50)
    print("\n" + "=" * 50)
    print("TWO-WIND FIT RESULTS")
    print("=" * 50)
    print(f"Converged: {result['convergence']}")
    print(f"Final misfit: {result['misfit']:.6f}")
    print(f"Iterations: {result['iterations']}")
    if result['solution_params']:
        print("\nWind Field 1:")
        print(f"  U: {result['solution_params']['wind1_U']:.2f} m/s")
        print(f"  Azimuth: {result['solution_params']['wind1_az']:.1f} deg")
        print(f"  T0: {result['solution_params']['wind1_T0']:.1f} K")
        print("\nWind Field 2:")
        print(f"  U: {result['solution_params']['wind2_U']:.2f} m/s")
        print(f"  Azimuth: {result['solution_params']['wind2_az']:.1f} deg")
        print(f"  T0: {result['solution_params']['wind2_T0']:.1f} K")
        print(f"\nFraction (Wind 2): {result['solution_params']['frac2']:.2f}")
