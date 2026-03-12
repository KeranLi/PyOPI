"""
Input/Output module for OPI.

Handles data loading, coordinate transformations, MATLAB compatibility,
run file parsing, and solutions file management.
"""

# Coordinate transformations
from .coordinates import lonlat2xy, xy2lonlat

# Data loading
from .data_loader import grid_read, tukey_window, estimate_mwl, get_input

# MATLAB compatibility
from .matlab_compat import (
    load_opi_results,
    save_opi_results,
    convert_run_file,
    compare_matlab_python_results
)

# Run file parsing
from .run_file import (
    parse_run_file,
    write_run_file,
    get_parameter_info,
    validate_run_data
)

# Solutions file handling
from .solutions_file import (
    SolutionsFileWriter,
    parse_solutions_file,
    get_best_solution,
    merge_solutions_files
)

__all__ = [
    # Coordinates
    'lonlat2xy',
    'xy2lonlat',
    # Data loader
    'grid_read',
    'tukey_window',
    'estimate_mwl',
    'get_input',
    # MATLAB compatibility
    'load_opi_results',
    'save_opi_results',
    'convert_run_file',
    'compare_matlab_python_results',
    # Run file
    'parse_run_file',
    'write_run_file',
    'get_parameter_info',
    'validate_run_data',
    # Solutions file
    'SolutionsFileWriter',
    'parse_solutions_file',
    'get_best_solution',
    'merge_solutions_files',
]
