"""
Two-Wind Parameter Fitting Function for OPI

This module implements parameter fitting for the two-wind model,
which optimizes 19 parameters for two distinct moisture sources.

This replicates the functionality of MATLAB's opiFit_TwoWinds.m
"""

import os
import numpy as np
from datetime import datetime
from scipy.io import loadmat, savemat
from matplotlib.path import Path

from .calc_two_winds import calc_two_winds
from .io.run_file import parse_run_file
from .io.data_loader import get_input
from .io.coordinates import lonlat2xy
from .fmin_crs3 import fmin_crs3
from .catchment_nodes import catchment_nodes
from .constants import SD_RES_RATIO, HR, TC2K


def load_continental_divide(data_path, divide_file, lon0, lat0):
    """
    Load continental divide data.
    
    Returns:
    --------
    cont_divide_x, cont_divide_y : array-like or None
        Divide coordinates in x,y space
    """
    if not divide_file:
        return None, None
    
    divide_path = os.path.join(data_path, divide_file)
    if not os.path.exists(divide_path):
        return None, None
    
    try:
        divide_data = loadmat(divide_path)
        cont_divide_lon = divide_data['contDivideLon'].flatten()
        cont_divide_lat = divide_data['contDivideLat'].flatten()
        
        # Convert to x,y coordinates
        cont_divide_x, cont_divide_y = lonlat2xy(cont_divide_lon, cont_divide_lat, lon0, lat0)
        
        return cont_divide_x, cont_divide_y
    except Exception as e:
        print(f"Warning: Failed to load continental divide: {e}")
        return None, None


def find_polygon_intersections(region_x, region_y, line_x, line_y):
    """
    Find intersections between a polygon and a line.
    Simplified implementation of MATLAB's polyxpoly.
    
    Returns:
    --------
    point_x, point_y : arrays
        Intersection points
    indices : list
        Edge indices for intersections
    """
    from shapely.geometry import Polygon, LineString, MultiLineString
    from shapely.ops import unary_union
    
    try:
        # Create polygon and line
        poly = Polygon(zip(region_x, region_y))
        line = LineString(zip(line_x, line_y))
        
        # Find intersection
        intersection = poly.boundary.intersection(line)
        
        points_x = []
        points_y = []
        
        if intersection.is_empty:
            return np.array([]), np.array([]), []
        
        if intersection.geom_type == 'Point':
            points_x = [intersection.x]
            points_y = [intersection.y]
        elif intersection.geom_type == 'MultiPoint':
            for point in intersection.geoms:
                points_x.append(point.x)
                points_y.append(point.y)
        
        return np.array(points_x), np.array(points_y), []
    except:
        # Fallback: simple line intersection
        return np.array([]), np.array([]), []


def determine_sample_sides(sample_x, sample_y, cont_divide_x, cont_divide_y,
                           azimuth_1, x, y):
    """
    Determine which side of continental divide each sample falls on.
    
    This implements the logic from MATLAB's opiCalc_TwoWinds.
    
    Parameters:
    -----------
    sample_x, sample_y : array-like
        Sample coordinates
    cont_divide_x, cont_divide_y : array-like
        Continental divide coordinates
    azimuth_1 : float
        Azimuth of wind field 1 (degrees)
    x, y : array-like
        Grid coordinates defining the region
    
    Returns:
    --------
    is_sample_side_01 : array-like
        Boolean array, True for samples on side 1
    """
    n_samples = len(sample_x)
    
    # Create region boundary
    region_x = np.array([x[0], x[-1], x[-1], x[0], x[0]])
    region_y = np.array([y[0], y[0], y[-1], y[-1], y[0]])
    
    # Find intersections between divide and region boundary
    point_x, point_y, _ = find_polygon_intersections(
        region_x, region_y, cont_divide_x, cont_divide_y
    )
    
    if len(point_x) < 2:
        # No valid intersections, assign all to side 1
        print("Warning: Continental divide has less than 2 intersections with region boundary")
        return np.ones(n_samples, dtype=bool)
    
    # Limit to first and last intersection
    if len(point_x) > 2:
        point_x = np.array([point_x[0], point_x[-1]])
        point_y = np.array([point_y[0], point_y[-1]])
    
    # Create polygon for side 1
    # This is simplified - full implementation would trace along region boundary
    
    # Calculate mean position
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    r = min([x[-1] - x[0], y[-1] - y[0]]) / 2
    
    # Wind entry point for state 1
    wind_entry_x = mean_x + r * np.sin(np.deg2rad(-azimuth_1))
    wind_entry_y = mean_y + r * np.cos(np.deg2rad(-azimuth_1))
    
    # Simple line-based classification
    # Divide line: ax + by + c = 0
    x1, y1 = cont_divide_x[0], cont_divide_y[0]
    x2, y2 = cont_divide_x[-1], cont_divide_y[-1]
    
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    
    # Side for each sample
    sample_dist = a * sample_x + b * sample_y + c
    
    # Side for wind entry
    wind_dist = a * wind_entry_x + b * wind_entry_y + c
    
    # Samples on same side as wind entry are for wind 1
    is_sample_side_01 = (sample_dist * wind_dist) > 0
    
    return is_sample_side_01


def opi_fit_two_winds(run_file_path=None, verbose=True, max_iterations=10000,
                      parallel=False):
    """
    Performs parameter fitting for the two-wind OPI model.
    
    This function estimates the optimal 19 parameters for the two-wind model
    using observational data and CRS3 global optimization.
    
    Parameters:
    -----------
    run_file_path : str
        Path to the run file containing model parameters and file paths
    verbose : bool, optional
        Whether to print detailed progress information
    max_iterations : int, optional
        Maximum number of optimization iterations
    parallel : bool, optional
        Whether to use parallel processing
    
    Returns:
    --------
    dict
        Dictionary containing fitted parameters and results
    """
    start_time = datetime.now()
    
    if verbose:
        print("OPI Two-Wind Parameter Fitting")
        print("=" * 50)
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load run file
    if run_file_path is None or not os.path.exists(run_file_path):
        raise ValueError("Run file path is required for opi_fit_two_winds")
    
    run_data = parse_run_file(run_file_path)
    run_path = run_data['run_path']
    run_filename = run_data['run_file']
    
    if verbose:
        print(f"Run file: {run_filename}")
        print(f"Run title: {run_data['run_title']}")
    
    # Validate
    if run_data['n_parameters'] != 19:
        raise ValueError(f"Expected 19 parameters for two-wind fit, got {run_data['n_parameters']}")
    
    if run_data.get('beta') is not None:
        raise ValueError("Run file includes a solution vector, which is not allowed for opi_fit_two_winds")
    
    if run_data.get('cont_divide_file') is None:
        raise ValueError("Continental divide file is required for two-wind fitting")
    
    # Load input data
    data_path = run_data['data_path']
    
    if run_data.get('sample_file') is None:
        raise ValueError("Sample file is required for fitting")
    
    if verbose:
        print(f"\nLoading input data...")
        print(f"Data path: {data_path}")
        print(f"Topography file: {run_data['topo_file']}")
        print(f"Sample file: {run_data['sample_file']}")
        print(f"Continental divide file: {run_data['cont_divide_file']}")
    
    input_data = get_input(
        data_path=data_path,
        topo_file=run_data['topo_file'],
        r_tukey=run_data['r_tukey'],
        sample_file=run_data['sample_file'],
        sd_res_ratio=SD_RES_RATIO
    )
    
    h_grid = input_data['h_grid']
    n_samples = len(input_data['sample_x'])
    
    if verbose:
        print(f"Grid size: {h_grid.shape[1]} x {h_grid.shape[0]}")
        print(f"Maximum elevation: {np.max(h_grid):.0f} m")
        print(f"Number of samples: {n_samples}")
    
    # Load continental divide
    cont_divide_x, cont_divide_y = load_continental_divide(
        data_path, run_data['cont_divide_file'],
        input_data['lon0'], input_data['lat0']
    )
    
    if cont_divide_x is None:
        raise ValueError("Failed to load continental divide file")
    
    # Calculate catchment nodes
    ij_catch, ptr_catch = catchment_nodes(
        input_data['sample_x'], input_data['sample_y'],
        input_data['sample_lc'], input_data['x'], input_data['y'],
        h_grid
    )
    
    # Initial guess for wind 1 azimuth (for side determination)
    # Will be updated during optimization
    initial_azimuth_1 = 90.0
    
    # Determine initial sample sides
    is_sample_side_01 = determine_sample_sides(
        input_data['sample_x'], input_data['sample_y'],
        cont_divide_x, cont_divide_y,
        initial_azimuth_1, input_data['x'], input_data['y']
    )
    
    if verbose:
        print(f"\nSample classification:")
        print(f"  Side 1 (Wind 1): {np.sum(is_sample_side_01)}")
        print(f"  Side 2 (Wind 2): {np.sum(~is_sample_side_01)}")
    
    # Prepare bounds
    bounds_lower = np.array(run_data['l_b'])
    bounds_upper = np.array(run_data['u_b'])
    param_bounds = [(bounds_lower[i], bounds_upper[i]) for i in range(19)]
    
    # Determine free parameters
    free_params = bounds_lower != bounds_upper
    n_free = np.sum(free_params)
    
    if n_free == 0:
        raise ValueError("No free parameters to fit!")
    
    # Initial guess
    initial_guess = np.where(free_params,
                           (bounds_lower + bounds_upper) / 2,
                           bounds_lower)
    
    if verbose:
        print(f"\nParameter fitting:")
        print(f"  Total parameters: 19")
        print(f"  Free parameters: {n_free}")
        print(f"  Fixed parameters: {19 - n_free}")
        print(f"  Optimization method: CRS3")
        print(f"  Max iterations: {max_iterations}")
    
    # Prepare covariance
    cov = input_data.get('cov', np.array([[1e-6, 0], [0, 1e-6]]))
    
    # Objective function
    def objective(beta_free):
        """Calculate total chi-square for two-wind model."""
        beta = bounds_lower.copy()
        beta[free_params] = beta_free
        
        # Extract azimuth 1 for side redetermination (optional)
        # azimuth_1 = beta[1]
        # Could re-determine sides here if needed
        
        try:
            result = calc_two_winds(
                beta=beta,
                f_c=input_data['f_c'],
                h_r=input_data.get('h_r', HR),
                x=input_data['x'],
                y=input_data['y'],
                lat=np.array([input_data['lat0']]),
                lat0=input_data['lat0'],
                h_grid=h_grid,
                b_mwl_sample=input_data['b_mwl_sample'],
                ij_catch=ij_catch,
                ptr_catch=ptr_catch,
                sample_d2h=input_data['sample_d2h'],
                sample_d18o=input_data['sample_d18o'],
                cov=cov,
                n_parameters_free=n_free,
                is_fit=True
            )
            
            chi_r2 = result[0]
            
            if np.isnan(chi_r2) or np.isinf(chi_r2):
                return 1e6
            
            return chi_r2
            
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
            epsilon=run_data.get('epsilon0', 1e-4),
            max_iter=max_iterations,
            random_state=42,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"  Final misfit: {opt_result.fun:.6f}")
            print(f"  Iterations: {opt_result.n_iter}")
            print(f"  Function evaluations: {opt_result.nfev}")
        
        # Reconstruct solution
        solution_vector = bounds_lower.copy()
        solution_vector[free_params] = opt_result.x
        
        # Final calculation
        final_result = calc_two_winds(
            beta=solution_vector,
            f_c=input_data['f_c'],
            h_r=input_data.get('h_r', HR),
            x=input_data['x'],
            y=input_data['y'],
            lat=np.array([input_data['lat0']]),
            lat0=input_data['lat0'],
            h_grid=h_grid,
            b_mwl_sample=input_data['b_mwl_sample'],
            ij_catch=ij_catch,
            ptr_catch=ptr_catch,
            sample_d2h=input_data['sample_d2h'],
            sample_d18o=input_data['sample_d18o'],
            cov=cov,
            n_parameters_free=n_free,
            is_fit=False
        )
        
        # Unpack
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
         fraction_p_grid) = final_result
        
        # Prepare output
        param_names = [
            'U_1', 'azimuth_1', 'T0_1', 'M_1', 'kappa_1', 'tau_c_1',
            'd2h0_1', 'd_d2h0_d_lat_1', 'f_p0_1',
            'U_2', 'azimuth_2', 'T0_2', 'M_2', 'kappa_2', 'tau_c_2',
            'd2h0_2', 'd_d2h0_d_lat_2', 'f_p0_2',
            'fraction'
        ]
        solution_params = dict(zip(param_names, solution_vector))
        
        derived_params = {
            'chi_r2': float(opt_result.fun),
            'degrees_of_freedom': int(nu),
            'n_samples_wet': int(np.sum(p_sum_pred > 0)),
            'function_evaluations': opt_result.nfev,
            'iterations': opt_result.n_iter
        }
        
        # Save results
        end_time = datetime.now()
        results_data = {
            'startTimeOpiFit': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'endTimeOpiFit': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'runPath': run_path,
            'runFile': run_filename,
            'runTitle': run_data['run_title'],
            'beta': solution_vector,
            'chiR2': float(opt_result.fun),
            'nu': int(nu),
            'pGrid': p_grid,
            'd2HGrid': d2h_grid,
            'd18OGrid': d18o_grid,
            'pGrid_1': p_grid_1,
            'pGrid_2': p_grid_2,
            'd2HGrid_1': d2h_grid_1,
            'd2HGrid_2': d2h_grid_2,
            'd18OGrid_1': d18o_grid_1,
            'd18OGrid_2': d18o_grid_2,
            'isSampleSide01': is_sample_side_01,
        }
        
        results_path = os.path.join(run_path, 'opiFit_TwoWinds_Results.mat')
        savemat(results_path, results_data)
        
        if verbose:
            print(f"\nResults saved to: {results_path}")
        
        return {
            'success': True,
            'solution_params': solution_params,
            'derived_params': derived_params,
            'misfit': float(opt_result.fun),
            'iterations': opt_result.n_iter,
            'convergence': opt_result.success,
            'message': 'Optimization completed successfully',
            'solution_vector': solution_vector.tolist(),
            'results_path': results_path,
            'is_sample_side_01': is_sample_side_01
        }
        
    except Exception as e:
        if verbose:
            print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'success': False,
            'solution_params': {},
            'misfit': np.nan,
            'convergence': False,
            'message': f'Optimization failed: {str(e)}',
            'results_path': None
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        result = opi_fit_two_winds(sys.argv[1], verbose=True, max_iterations=1000)
    else:
        print("Usage: python opi_fit_two_winds.py <run_file>")
        print("\nExample:")
        print("  python opi_fit_two_winds.py runs/run001/run001.run")
