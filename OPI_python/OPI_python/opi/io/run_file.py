"""
Run File Parser for OPI

Parses MATLAB-style run files for OPI configuration.
Full implementation matching MATLAB's getRunFile.m
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any


def parse_run_file(run_file_path: str) -> Dict[str, Any]:
    """
    Parse an OPI run file (full implementation).
    
    Matches MATLAB's getRunFile function behavior.
    
    Parameters
    ----------
    run_file_path : str
        Path to the .run file
    
    Returns
    -------
    run_data : dict
        Dictionary containing all run parameters:
        - run_path: directory containing run file
        - run_file: filename
        - run_title: title string
        - is_parallel: parallel mode flag (0 or 1)
        - data_path: data directory path
        - topo_file: topography filename
        - r_tukey: Tukey window fraction (0-1)
        - sample_file: sample data filename (or None)
        - cont_divide_file: continental divide file (or None)
        - restart_file: restart file (or None)
        - map_limits: [min_lon, max_lon, min_lat, max_lat]
        - section_lon0: section origin longitude (or None for 'map')
        - section_lat0: section origin latitude (or None for 'map')
        - mu: CRS3 population factor
        - epsilon0: CRS3 stopping criterion
        - parameter_labels: list of parameter label strings
        - exponents: power-of-10 scaling factors for parameters
        - l_b: lower bounds for parameters
        - u_b: upper bounds for parameters
        - beta: solution vector (if present in file)
    """
    if not os.path.exists(run_file_path):
        raise FileNotFoundError(f"Run file not found: {run_file_path}")
    
    run_path = os.path.dirname(os.path.abspath(run_file_path))
    run_filename = os.path.basename(run_file_path)
    
    with open(run_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove comments and empty lines, strip whitespace
    content_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('%'):
            content_lines.append(line)
    
    if len(content_lines) < 10:
        raise ValueError("Run file has insufficient content")
    
    idx = 0
    
    # Run title
    run_title = content_lines[idx]; idx += 1
    
    # Parallel mode (0 or 1)
    parallel_str = content_lines[idx]; idx += 1
    is_parallel = bool(int(parallel_str))
    
    # Data paths (2 lines - primary and alternate)
    data_path_primary = content_lines[idx]; idx += 1
    data_path_alt = content_lines[idx]; idx += 1
    
    # Use primary if it exists, otherwise alternate
    if os.path.isdir(data_path_primary):
        data_path = data_path_primary
    elif data_path_primary.lower() != 'no' and os.path.isdir(data_path_primary):
        data_path = data_path_primary
    elif data_path_alt.lower() != 'no' and os.path.isdir(data_path_alt):
        data_path = data_path_alt
    else:
        # Default to run_path/data
        data_path = os.path.join(run_path, 'data')
    
    # Remove trailing slashes
    data_path = data_path.rstrip('/\\')
    
    # Topography file
    topo_file = content_lines[idx]; idx += 1
    
    # Tukey window size (0-1)
    r_tukey = float(content_lines[idx]); idx += 1
    if not (0 <= r_tukey <= 1):
        raise ValueError("r_tukey must be between 0 and 1")
    
    # Sample file ('no' means no samples)
    sample_file = content_lines[idx]; idx += 1
    if sample_file.lower() == 'no':
        sample_file = None
    
    # Continental divide file
    cont_divide_file = content_lines[idx]; idx += 1
    if cont_divide_file.lower() == 'no':
        cont_divide_file = None
    
    # Map limits [min_lon, max_lon, min_lat, max_lat]
    map_limits_str = content_lines[idx]; idx += 1
    map_limits = [float(x) for x in re.findall(r'-?\d+\.?\d*', map_limits_str)]
    if len(map_limits) != 4:
        raise ValueError("Map limits must have 4 values")
    
    # Section origin
    section_origin_str = content_lines[idx]; idx += 1
    if section_origin_str.lower() == 'map':
        section_lon0 = None
        section_lat0 = None
    else:
        coords = [float(x) for x in re.findall(r'-?\d+\.?\d*', section_origin_str)]
        if len(coords) != 2:
            raise ValueError("Section origin must have 2 values or be 'map'")
        section_lon0 = coords[0]
        section_lat0 = coords[1]
    
    # CRS3 parameters [mu, epsilon0]
    crs_params_str = content_lines[idx]; idx += 1
    crs_params = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', crs_params_str)]
    if len(crs_params) != 2:
        raise ValueError("CRS3 parameters must have 2 values")
    mu = crs_params[0]
    epsilon0 = crs_params[1]
    
    # Restart file
    restart_file = content_lines[idx]; idx += 1
    if restart_file.lower() == 'no':
        restart_file = None
    
    # Parameter labels (pipe-separated)
    parameter_labels_str = content_lines[idx]; idx += 1
    parameter_labels = [label.strip() for label in parameter_labels_str.split('|')]
    n_parameters = len(parameter_labels)
    
    if n_parameters not in [9, 19]:
        raise ValueError("Number of parameters must be 9 or 19")
    
    # Exponents for power-of-10 scaling
    exponents_str = content_lines[idx]; idx += 1
    exponents = [int(x) for x in re.findall(r'-?\d+', exponents_str)]
    if len(exponents) != n_parameters:
        raise ValueError(f"Expected {n_parameters} exponents, got {len(exponents)}")
    
    # Lower bounds
    l_b_str = content_lines[idx]; idx += 1
    l_b = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', l_b_str)]
    if len(l_b) != n_parameters:
        raise ValueError(f"Expected {n_parameters} lower bounds, got {len(l_b)}")
    
    # Upper bounds
    u_b_str = content_lines[idx]; idx += 1
    u_b = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', u_b_str)]
    if len(u_b) != n_parameters:
        raise ValueError(f"Expected {n_parameters} upper bounds, got {len(u_b)}")
    
    # Validate bounds
    for i, (lb, ub) in enumerate(zip(l_b, u_b)):
        if lb > ub:
            raise ValueError(f"Lower bound greater than upper bound for parameter {i+1}")
    
    # Solution vector (beta) - optional, at end of file
    beta = None
    if idx < len(content_lines):
        beta_str = content_lines[idx]; idx += 1
        try:
            beta = [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e[+-]?\d+)?', beta_str)]
            if len(beta) != n_parameters:
                beta = None
        except:
            beta = None
    
    return {
        'run_path': run_path,
        'run_file': run_filename,
        'run_title': run_title,
        'is_parallel': is_parallel,
        'data_path': data_path,
        'topo_file': topo_file,
        'r_tukey': r_tukey,
        'sample_file': sample_file,
        'cont_divide_file': cont_divide_file,
        'restart_file': restart_file,
        'map_limits': map_limits,
        'section_lon0': section_lon0,
        'section_lat0': section_lat0,
        'mu': mu,
        'epsilon0': epsilon0,
        'parameter_labels': parameter_labels,
        'exponents': exponents,
        'l_b': l_b,
        'u_b': u_b,
        'beta': beta,
        'n_parameters': n_parameters,
        'n_free': sum(1 for lb, ub in zip(l_b, u_b) if lb != ub),
    }


def write_run_file(output_path: str, run_data: Dict[str, Any]) -> None:
    """
    Write an OPI run file.
    
    Parameters
    ----------
    output_path : str
        Path for output .run file
    run_data : dict
        Dictionary with run parameters (see parse_run_file for format)
    """
    lines = []
    lines.append(f"{run_data['run_title']}")
    lines.append(f"{int(run_data['is_parallel'])}")
    lines.append(f"{run_data['data_path']}")
    lines.append("no")  # Alternate data path
    lines.append(f"{run_data['topo_file']}")
    lines.append(f"{run_data['r_tukey']}")
    lines.append(f"{run_data['sample_file'] or 'no'}")
    lines.append(f"{run_data['cont_divide_file'] or 'no'}")
    lines.append(f"[{run_data['map_limits'][0]} {run_data['map_limits'][1]} {run_data['map_limits'][2]} {run_data['map_limits'][3]}]")
    
    if run_data['section_lon0'] is None:
        lines.append("map")
    else:
        lines.append(f"{run_data['section_lon0']} {run_data['section_lat0']}")
    
    lines.append(f"{run_data['mu']} {run_data['epsilon0']}")
    lines.append(f"{run_data['restart_file'] or 'no'}")
    lines.append("|".join(run_data['parameter_labels']))
    lines.append(" ".join(str(e) for e in run_data['exponents']))
    lines.append(" ".join(str(b) for b in run_data['l_b']))
    lines.append(" ".join(str(b) for b in run_data['u_b']))
    
    if run_data.get('beta') is not None:
        lines.append(" ".join(str(b) for b in run_data['beta']))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def get_parameter_info(param_idx: int, run_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about a specific parameter.
    
    Parameters
    ----------
    param_idx : int
        Parameter index (0-based)
    run_data : dict
        Run data dictionary
    
    Returns
    -------
    info : dict
        Parameter information
    """
    return {
        'index': param_idx,
        'label': run_data['parameter_labels'][param_idx],
        'exponent': run_data['exponents'][param_idx],
        'lower_bound': run_data['l_b'][param_idx],
        'upper_bound': run_data['u_b'][param_idx],
        'is_free': run_data['l_b'][param_idx] != run_data['u_b'][param_idx],
        'scale_factor': 10 ** (-run_data['exponents'][param_idx]),
    }


def validate_run_data(run_data: Dict[str, Any]) -> List[str]:
    """
    Validate run data and return list of errors.
    
    Parameters
    ----------
    run_data : dict
        Run data dictionary
    
    Returns
    -------
    errors : list
        List of error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    required = ['run_title', 'data_path', 'topo_file', 'l_b', 'u_b']
    for field in required:
        if field not in run_data:
            errors.append(f"Missing required field: {field}")
    
    # Check bounds
    if 'l_b' in run_data and 'u_b' in run_data:
        if len(run_data['l_b']) != len(run_data['u_b']):
            errors.append("Lower and upper bounds have different lengths")
        else:
            for i, (lb, ub) in enumerate(zip(run_data['l_b'], run_data['u_b'])):
                if lb > ub:
                    errors.append(f"Parameter {i+1}: lower bound > upper bound")
    
    # Check r_tukey
    if 'r_tukey' in run_data:
        if not (0 <= run_data['r_tukey'] <= 1):
            errors.append("r_tukey must be between 0 and 1")
    
    # Check data path exists
    if 'data_path' in run_data:
        if not os.path.isdir(run_data['data_path']):
            errors.append(f"Data path does not exist: {run_data['data_path']}")
    
    return errors
