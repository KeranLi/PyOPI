"""
OPI Simulation Runner

Main entry point for running OPI simulations from .run files.
Equivalent to MATLAB's opiCalc_OneWind and opiCalc_TwoWinds functions.
"""

import os
import sys
import numpy as np
from datetime import datetime
from scipy.io import savemat

from .app.calc_two_winds import opi_calc_two_winds
from .app.calc_one_wind import opi_calc_one_wind
from .io.run_file import parse_run_file
from .constants import HR, SD_RES_RATIO


def run_simulation(run_file_path, verbose=True):
    """
    Run OPI simulation from a .run file.
    
    Parameters:
    -----------
    run_file_path : str
        Path to the .run configuration file
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'success': bool indicating if simulation completed
        - 'results_path': path to saved results file
        - 'error': error message if failed
    """
    try:
        # Parse run file
        if verbose:
            print(f"Loading run file: {run_file_path}")
        
        run_params = parse_run_file(run_file_path)
        
        # Determine if one-wind or two-wind based on beta vector length
        beta = run_params.get('beta', [])
        
        if len(beta) == 9:
            model_type = 'one_wind'
        elif len(beta) == 19:
            model_type = 'two_winds'
        else:
            return {
                'success': False,
                'error': f'Invalid beta vector length: {len(beta)} (expected 9 or 19)'
            }
        
        if verbose:
            print(f"Detected model type: {model_type}")
            print(f"Run title: {run_params.get('title', 'N/A')}")
        
        # Get output directory
        run_dir = os.path.dirname(run_file_path) if os.path.dirname(run_file_path) else os.getcwd()
        
        # Run appropriate calculation
        if model_type == 'one_wind':
            result = opi_calc_one_wind(
                run_file_path=run_file_path,
                solution_vector=np.array(beta),
                verbose=verbose
            )
            results_filename = 'opiCalc_OneWind_Results.mat'
        else:
            result = opi_calc_two_winds(
                run_file_path=run_file_path,
                solution_vector=np.array(beta),
                verbose=verbose
            )
            results_filename = 'opiCalc_TwoWinds_Results.mat'
        
        # Save results
        results_path = os.path.join(run_dir, results_filename)
        
        # Prepare data for saving
        save_data = {
            'startTimeOpiCalc': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'runPath': run_dir,
            'runFile': os.path.basename(run_file_path),
            'runTitle': run_params.get('title', ''),
            'beta': np.array(beta),
            'x': result.get('x'),
            'y': result.get('y'),
            'hGrid': result.get('h_grid'),
            'mapLimits': run_params.get('map_limits', [0, 1, 0, 1]),
            'sectionLon0': run_params.get('section_lon0', np.mean(result.get('x', [0]))),
            'sectionLat0': run_params.get('section_lat0', np.mean(result.get('y', [0]))),
        }
        
        # Add results based on model type
        if model_type == 'two_winds':
            save_data.update({
                'pGrid_1': result['precipitation1'],
                'pGrid_2': result['precipitation2'],
                'pGrid': result['precipitation'],
                'd2HGrid_1': result['isotope1'],
                'd2HGrid_2': result['isotope2'],
                'd2HGrid': result['isotope'],
                'd18OGrid': result.get('d18o_grid'),
                'd18OGrid_1': result.get('d18o_grid_1'),
                'd18OGrid_2': result.get('d18o_grid_2'),
                'fMGrid_1': result.get('f_m_grid_1'),
                'fMGrid_2': result.get('f_m_grid_2'),
                'rHGrid_1': result.get('r_h_grid_1'),
                'rHGrid_2': result.get('r_h_grid_2'),
                'TGrid_1': result.get('T_grid_1'),  # Surface temperature
                'TGrid_2': result.get('T_grid_2'),
                'fractionPGrid': result.get('fraction_p_grid'),  # Fig 09
            })
        else:
            save_data.update({
                'pGrid': result.get('precipitation', result.get('p_grid')),
                'd2HGrid': result.get('isotope', result.get('d2h_grid')),
            })
        
        savemat(results_path, save_data)
        
        if verbose:
            print(f"\nResults saved to: {results_path}")
        
        return {
            'success': True,
            'results_path': results_path,
            'model_type': model_type
        }
        
    except Exception as e:
        import traceback
        if verbose:
            print(f"Error during simulation: {e}")
            traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e)
        }


if __name__ == "__main__":
    # Simple command-line interface
    if len(sys.argv) < 2:
        print("Usage: python -m opi.run_opi_simulation <run_file>")
        sys.exit(1)
    
    run_file = sys.argv[1]
    result = run_simulation(run_file, verbose=True)
    
    if result['success']:
        print("\nSimulation completed successfully!")
        sys.exit(0)
    else:
        print(f"\nSimulation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
