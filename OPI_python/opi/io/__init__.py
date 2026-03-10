"""
Input/Output Utilities for OPI

This module provides functions for loading and saving OPI data including:
- Topography grids (ASCII, NetCDF, GeoTIFF)
- Sample data (Excel, CSV)
- Run files (parameter configurations)
- Results (numpy arrays, MATLAB files)
"""

from .data_loader import (
    load_topography,
    load_samples_csv,
    save_results,
    load_results
)
from .run_file import parse_run_file

__all__ = [
    'load_topography',
    'load_samples_csv', 
    'save_results',
    'load_results',
    'parse_run_file',
]
