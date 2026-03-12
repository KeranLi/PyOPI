"""
OPI (Orographic Precipitation and Isotopes) - Python Package

This package provides computational tools for the analysis of precipitation 
and isotope fractionation associated with steady atmospheric flow over 
an arbitrary three-dimensional topography.

This Python implementation replicates the functionality of the MATLAB OPI 3.7 model
developed by Mark Brandon at Yale University.

Main Functions:
---------------
- opi_calc_one_wind : Calculate results for one-wind model
- opi_calc_two_winds : Calculate results for two-wind model
- opi_fit_one_wind : Fit parameters for one-wind model
- opi_fit_two_winds : Fit parameters for two-wind model
- opi_maps_one_wind : Generate maps for one-wind results
- opi_maps_two_winds : Generate maps for two-wind results
- run_simulation : Main entry point for running simulations from .run files

Example Usage:
--------------
    # Run a simulation from a .run file
    from opi import run_simulation
    results = run_simulation('runs/run001/run001.run')
    
    # Or use the module directly
    from opi import opi_calc_one_wind
    result = opi_calc_one_wind('runs/run001/run001.run')
    
    # Generate maps from results
    from opi import opi_maps_one_wind
    opi_maps_one_wind('runs/run001/opiCalc_OneWind_Results.mat')

For more information, see README.md and READMEMAT.md
"""

__version__ = "1.0.0"
__author__ = "Python implementation based on MATLAB code by Mark Brandon (Yale University)"

# Import main simulation functions
from .run_opi_simulation import run_simulation

# Import calculation functions
from .opi_calc_one_wind import opi_calc_one_wind
from .opi_calc_two_winds import opi_calc_two_winds
from .calc_one_wind import calc_one_wind
from .calc_two_winds import calc_two_winds

# Import fitting functions
from .opi_fit_one_wind import opi_fit_one_wind
from .opi_fit_two_winds import opi_fit_two_winds

# Import visualization functions
from .opi_maps_one_wind import opi_maps_one_wind
from .opi_maps_two_winds import opi_maps_two_winds

# Import core utilities
from .base_state import base_state
from .saturated_vapor_pressure import saturated_vapor_pressure
from .coordinates import lonlat2xy, xy2lonlat
from .wind_path import wind_path
from .catchment_nodes import catchment_nodes
from .catchment_indices import catchment_indices
from .precipitation_grid import precipitation_grid
from .isotope_grid import isotope_grid
from .fractionation_hydrogen import fractionation_hydrogen
from .fractionation_oxygen import fractionation_oxygen

# Import I/O utilities
from .io.run_file import parse_run_file, write_run_file
from .io.data_loader import get_input, grid_read

# Import optimization
from .fmin_crs3 import fmin_crs3, CRS3Optimizer

# Import constants
from .constants import (
    G, CPD, CPV, RD, RV, L, P0, EPSILON, GAMMA_D,
    RADIUS_EARTH, M_PER_DEGREE, OMEGA,
    TC2K, HR, SD_RES_RATIO
)

# Define public API
__all__ = [
    # Main entry point
    'run_simulation',
    
    # High-level calculation functions
    'opi_calc_one_wind',
    'opi_calc_two_winds',
    
    # Core calculation functions
    'calc_one_wind',
    'calc_two_winds',
    
    # Fitting functions
    'opi_fit_one_wind',
    'opi_fit_two_winds',
    
    # Visualization functions
    'opi_maps_one_wind',
    'opi_maps_two_winds',
    
    # Core utilities
    'base_state',
    'saturated_vapor_pressure',
    'lonlat2xy',
    'xy2lonlat',
    'wind_path',
    'catchment_nodes',
    'catchment_indices',
    'precipitation_grid',
    'isotope_grid',
    'fractionation_hydrogen',
    'fractionation_oxygen',
    
    # I/O utilities
    'parse_run_file',
    'write_run_file',
    'get_input',
    'grid_read',
    
    # Optimization
    'fmin_crs3',
    'CRS3Optimizer',
    
    # Constants
    'G', 'CPD', 'CPV', 'RD', 'RV', 'L', 'P0', 'EPSILON', 'GAMMA_D',
    'RADIUS_EARTH', 'M_PER_DEGREE', 'OMEGA',
    'TC2K', 'HR', 'SD_RES_RATIO'
]


def print_version():
    """Print version information."""
    print(f"OPI (Orographic Precipitation and Isotopes) Python Package")
    print(f"Version: {__version__}")
    print(f"Based on MATLAB OPI 3.7 by Mark Brandon, Yale University")
