"""
Run file parsing for OPI.

This module handles parsing of OPI run files which define
simulation parameters, file paths, and sample data.
"""

import os
import numpy as np
import pandas as pd


def parse_run_file(run_file_path):
    """
    Parse an OPI run file.
    
    Run files define simulation parameters, topography files, sample data,
    and various settings for OPI calculations.
    
    Parameters
    ----------
    run_file_path : str
        Path to the run file
        
    Returns
    -------
    run_data : dict
        Dictionary containing all parsed run file parameters
    """
    if not os.path.exists(run_file_path):
        raise FileNotFoundError(f"Run file not found: {run_file_path}")
    
    run_data = {
        'path_name': '',
        'topo_file': '',
        'sample_file': '',
        'save_name': '',
        'tukey': 0.0,
        'rayleigh': False,
        'sample_data': None,
        'n_samples': 0,
        'param_names': [],
        'param_init': [],
        'param_min': [],
        'param_max': [],
        'param_fit': [],
        'sample_x': [],
        'sample_y': [],
        'sample_lc': [],
        'sample_d2h': [],
        'sample_d18o': [],
        'sample_dxs': [],
    }
    
    # Get the directory containing the run file
    run_dir = os.path.dirname(os.path.abspath(run_file_path))
    
    with open(run_file_path, 'r') as f:
        lines = f.readlines()
    
    section = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        
        # Skip empty lines and comments
        if not line or line.startswith('%') or line.startswith('#'):
            continue
        
        # Parse section headers
        if line.startswith('[') and line.endswith(']'):
            section = line[1:-1].lower()
            continue
        
        # Parse key-value pairs
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Remove comments from value
            if '%' in value:
                value = value.split('%')[0].strip()
            if '#' in value:
                value = value.split('#')[0].strip()
            
            # Parse based on key
            if key == 'path_name':
                run_data['path_name'] = os.path.join(run_dir, value)
            elif key == 'topo_file':
                run_data['topo_file'] = value
            elif key == 'sample_file':
                run_data['sample_file'] = value
            elif key == 'save_name':
                run_data['save_name'] = value
            elif key == 'tukey':
                run_data['tukey'] = float(value)
            elif key == 'rayleigh':
                run_data['rayleigh'] = value.lower() in ['true', '1', 'yes', 'on']
    
    # Try to load sample file if specified
    if run_data['sample_file']:
        sample_path = os.path.join(run_data['path_name'], run_data['sample_file'])
        if os.path.exists(sample_path):
            try:
                run_data['sample_data'] = pd.read_csv(sample_path)
                run_data['n_samples'] = len(run_data['sample_data'])
                
                # Extract sample coordinates and isotope values
                df = run_data['sample_data']
                if 'x' in df.columns:
                    run_data['sample_x'] = df['x'].values
                if 'y' in df.columns:
                    run_data['sample_y'] = df['y'].values
                if 'lc' in df.columns:
                    run_data['sample_lc'] = df['lc'].values
                if 'd2h' in df.columns:
                    run_data['sample_d2h'] = df['d2h'].values
                if 'd18o' in df.columns:
                    run_data['sample_d18o'] = df['d18o'].values
                if 'dxs' in df.columns:
                    run_data['sample_dxs'] = df['dxs'].values
            except Exception as e:
                print(f"Warning: Could not load sample file: {e}")
    
    return run_data
