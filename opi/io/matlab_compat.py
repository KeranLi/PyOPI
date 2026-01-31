"""
MATLAB Compatibility Module

Provides functions for reading and writing MATLAB .mat files
in formats compatible with OPI MATLAB version.
"""

import numpy as np
from scipy.io import loadmat, savemat
import os


def load_opi_results(mat_file_path):
    """
    Load OPI results from MATLAB .mat file.
    
    Parameters
    ----------
    mat_file_path : str
        Path to MATLAB .mat file
    
    Returns
    -------
    results : dict
        Dictionary with OPI results
    """
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"MAT file not found: {mat_file_path}")
    
    # Load the mat file
    mat_data = loadmat(mat_file_path, squeeze_me=True, struct_as_record=False)
    
    # Extract variables
    results = {}
    
    # Map MATLAB variable names to Python-friendly names
    variable_map = {
        'lon': 'lon',
        'lat': 'lat',
        'x': 'x',
        'y': 'y',
        'hGrid': 'h_grid',
        'lon0': 'lon0',
        'lat0': 'lat0',
        'pGrid': 'p_grid',
        'd2HGrid': 'd2h_grid',
        'd18OGrid': 'd18o_grid',
        'fMGrid': 'f_m_grid',
        'rHGrid': 'r_h_grid',
        'beta': 'beta',
        'chiR2': 'chi_r2',
        'nu': 'nu',
        'stdResiduals': 'std_residuals',
        'zBar': 'z_bar',
        'T': 'T',
        'gammaEnv': 'gamma_env',
        'gammaSat': 'gamma_sat',
        'gammaRatio': 'gamma_ratio',
        'rhoS0': 'rho_s0',
        'hS': 'h_s',
        'rho0': 'rho0',
        'hRho': 'h_rho',
        'tauF': 'tau_f',
        'sampleLon': 'sample_lon',
        'sampleLat': 'sample_lat',
        'sampleX': 'sample_x',
        'sampleY': 'sample_y',
        'sampleD2H': 'sample_d2h',
        'sampleD18O': 'sample_d18o',
        'sampleDExcess': 'sample_d_excess',
        'sampleLC': 'sample_lc',
        'd2HPred': 'd2h_pred',
        'd18OPred': 'd18o_pred',
        'ijCatch': 'ij_catch',
        'ptrCatch': 'ptr_catch',
        'runPath': 'run_path',
        'runFile': 'run_file',
        'runTitle': 'run_title',
        'dataPath': 'data_path',
        'topoFile': 'topo_file',
        'rTukey': 'r_tukey',
        'sampleFile': 'sample_file',
        'hR': 'h_r',
        'sdResRatio': 'sd_res_ratio',
        'fC': 'f_c',
    }
    
    for mat_name, py_name in variable_map.items():
        if mat_name in mat_data:
            value = mat_data[mat_name]
            # Convert numpy arrays to appropriate types
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    value = value.item()
            results[py_name] = value
    
    return results


def save_opi_results(output_path, results_dict):
    """
    Save OPI results to MATLAB .mat file.
    
    Parameters
    ----------
    output_path : str
        Path to output .mat file
    results_dict : dict
        Dictionary with OPI results
    
    Returns
    -------
    success : bool
        True if saved successfully
    """
    # Map Python variable names to MATLAB names
    mat_dict = {}
    
    reverse_map = {
        'lon': 'lon',
        'lat': 'lat',
        'x': 'x',
        'y': 'y',
        'h_grid': 'hGrid',
        'lon0': 'lon0',
        'lat0': 'lat0',
        'p_grid': 'pGrid',
        'd2h_grid': 'd2HGrid',
        'd18o_grid': 'd18OGrid',
        'f_m_grid': 'fMGrid',
        'r_h_grid': 'rHGrid',
        'beta': 'beta',
        'chi_r2': 'chiR2',
        'nu': 'nu',
        'std_residuals': 'stdResiduals',
        'z_bar': 'zBar',
        'T': 'T',
        'gamma_env': 'gammaEnv',
        'gamma_sat': 'gammaSat',
        'gamma_ratio': 'gammaRatio',
        'rho_s0': 'rhoS0',
        'h_s': 'hS',
        'rho0': 'rho0',
        'h_rho': 'hRho',
        'tau_f': 'tauF',
        'sample_lon': 'sampleLon',
        'sample_lat': 'sampleLat',
        'sample_x': 'sampleX',
        'sample_y': 'sampleY',
        'sample_d2h': 'sampleD2H',
        'sample_d18o': 'sampleD18O',
        'sample_d_excess': 'sampleDExcess',
        'sample_lc': 'sampleLC',
        'd2h_pred': 'd2HPred',
        'd18o_pred': 'd18OPred',
        'ij_catch': 'ijCatch',
        'ptr_catch': 'ptrCatch',
        'run_path': 'runPath',
        'run_file': 'runFile',
        'run_title': 'runTitle',
        'data_path': 'dataPath',
        'topo_file': 'topoFile',
        'r_tukey': 'rTukey',
        'sample_file': 'sampleFile',
        'h_r': 'hR',
        'sd_res_ratio': 'sdResRatio',
        'f_c': 'fC',
    }
    
    for py_name, mat_name in reverse_map.items():
        if py_name in results_dict:
            mat_dict[mat_name] = results_dict[py_name]
    
    # Ensure arrays are in Fortran order (MATLAB convention)
    for key in mat_dict:
        if isinstance(mat_dict[key], np.ndarray):
            if mat_dict[key].ndim >= 2:
                mat_dict[key] = np.asfortranarray(mat_dict[key])
    
    # Save to file
    savemat(output_path, mat_dict, do_compression=True)
    
    return True


def convert_run_file(run_file_path):
    """
    Convert MATLAB run file to Python-compatible dictionary.
    
    MATLAB run files are text files with specific line ordering.
    
    Parameters
    ----------
    run_file_path : str
        Path to run file
    
    Returns
    -------
    run_data : dict
        Dictionary with run parameters
    """
    if not os.path.exists(run_file_path):
        raise FileNotFoundError(f"Run file not found: {run_file_path}")
    
    with open(run_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Parse based on MATLAB format
    run_data = {
        'run_title': lines[0] if len(lines) > 0 else 'Run',
        'parallel_mode': int(lines[1]) if len(lines) > 1 else 0,
        'data_path': lines[2] if len(lines) > 2 else 'data',
        'aux_path': lines[3] if len(lines) > 3 else '',
        'topography_file': lines[4] if len(lines) > 4 else 'topography.mat',
        'r_tukey': float(lines[5]) if len(lines) > 5 else 0.0,
        'sample_file': lines[6] if len(lines) > 6 else 'no',
        'cont_divide_file': lines[7] if len(lines) > 7 else 'no',
        'restart_file': lines[8] if len(lines) > 8 else 'no',
    }
    
    # Parse map limits (line 9)
    if len(lines) > 8:
        try:
            map_limits = [float(x) for x in lines[8].strip('[]').split()]
            run_data['map_limits'] = map_limits
        except:
            run_data['map_limits'] = [0, 0, 0, 0]
    
    # Path origin (line 10)
    run_data['path_origin'] = lines[9] if len(lines) > 9 else 'map'
    
    # Section coordinates (lines 11-12)
    run_data['section_lon0'] = float(lines[10]) if len(lines) > 10 else 0.0
    run_data['section_lat0'] = float(lines[11]) if len(lines) > 11 else 0.0
    
    # Optimization parameters (lines 13-14)
    run_data['mu'] = int(lines[12]) if len(lines) > 12 else 25
    run_data['epsilon'] = float(lines[13]) if len(lines) > 13 else 1e-6
    
    # Parameter labels and bounds (remaining lines)
    # This is more complex and would need full parsing
    
    return run_data


def compare_matlab_python_results(matlab_file, python_file, tol=1e-6):
    """
    Compare results from MATLAB and Python runs.
    
    Parameters
    ----------
    matlab_file : str
        Path to MATLAB results .mat file
    python_file : str
        Path to Python results .mat file
    tol : float
        Tolerance for comparison
    
    Returns
    -------
    comparison : dict
        Dictionary with comparison results
    """
    mat_results = load_opi_results(matlab_file)
    py_results = load_opi_results(python_file)
    
    comparison = {
        'matches': {},
        'mismatches': {},
        'missing_in_matlab': [],
        'missing_in_python': []
    }
    
    # Find common keys
    mat_keys = set(mat_results.keys())
    py_keys = set(py_results.keys())
    
    common_keys = mat_keys & py_keys
    comparison['missing_in_matlab'] = list(py_keys - mat_keys)
    comparison['missing_in_python'] = list(mat_keys - py_keys)
    
    for key in common_keys:
        mat_val = mat_results[key]
        py_val = py_results[key]
        
        if isinstance(mat_val, np.ndarray) and isinstance(py_val, np.ndarray):
            if mat_val.shape == py_val.shape:
                max_diff = np.max(np.abs(mat_val - py_val))
                matches = max_diff < tol
            else:
                matches = False
                max_diff = None
        elif isinstance(mat_val, (int, float)) and isinstance(py_val, (int, float)):
            max_diff = abs(mat_val - py_val)
            matches = max_diff < tol
        else:
            matches = str(mat_val) == str(py_val)
            max_diff = None
        
        if matches:
            comparison['matches'][key] = max_diff
        else:
            comparison['mismatches'][key] = max_diff
    
    return comparison
