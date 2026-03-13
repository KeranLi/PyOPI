"""
OPI (Orographic Precipitation and Isotopes) - Python Package

This package provides computational tools for the analysis of precipitation 
and isotope fractionation associated with steady atmospheric flow over 
an arbitrary three-dimensional topography.

This Python implementation replicates the functionality of the MATLAB OPI 3.7 model
developed by Mark Brandon at Yale University.

Package Structure:
------------------
- opi.physics: Physical model calculations (base_state, precipitation, isotopes)
- opi.app: High-level application functions (calc_one_wind, calc_two_winds, fitting)
- opi.viz: Visualization tools (maps, plots)
- opi.io: Input/Output utilities (data_loader, run_file, coordinates)
- opi.tools: Utility tools (synthetic topography, climate records)

Example Usage:
--------------
    # Run a one-wind calculation
    from opi.app import opi_calc_one_wind
    result = opi_calc_one_wind(run_file_path)
    
    # Run a two-wind calculation
    from opi.app import opi_calc_two_winds
    result = opi_calc_two_winds(run_file_path)
    
    # Generate maps
    from opi.viz import maps
    maps.plot_results(result)

For more information, see README.md and docs/
"""

__version__ = "1.0.0"
__author__ = "Python implementation based on MATLAB code by Mark Brandon (Yale University)"

# Core physical calculations
from .physics import (
    # Thermodynamics
    base_state,
    saturated_vapor_pressure,
    # Fourier solution
    wind_grid,
    fourier_solution,
    # Precipitation and isotopes
    precipitation_grid,
    isotope_grid,
    # Fractionation
    fractionation_hydrogen,
    fractionation_oxygen,
    # Core calculations
    calc_one_wind,
    calc_two_winds,
)

# Coordinate utilities
from .coordinates import lonlat2xy, xy2lonlat
from .wind_path import wind_path
from .catchment import catchment_nodes, catchment_indices

# App-level functions
from .app import (
    opi_calc_one_wind,
    opi_calc_two_winds,
    opi_fit_one_wind,
    opi_fit_two_winds,
    opi_plots_one_wind,
)

# Visualization modules
from .viz import (
    maps,
    plots,
    advanced,
    export,
)

# I/O utilities
from .io import (
    parse_run_file,
    write_run_file,
    grid_read,
    get_input,
    lonlat2xy as convert_coordinates,
)

# Optimization
from .optimization import (
    fmin_crs3,
    CRS3Optimizer,
)

# Constants
from .constants import (
    G, CPD, CPV, RD, RV, L, P0, EPSILON, GAMMA_D,
    RADIUS_EARTH, M_PER_DEGREE, OMEGA,
    TC2K, HR, SD_RES_RATIO
)

# Utilities
from .utils import (
    deuterium_excess,
    meteoric_water_line,
    wind_components,
    mountain_height_number,
    rossby_number,
    froude_number,
    summarize_results
)


# Define public API
__all__ = [
    # Version
    '__version__',
    
    # Core physical functions
    'base_state',
    'saturated_vapor_pressure',
    'wind_grid',
    'fourier_solution',
    'precipitation_grid',
    'isotope_grid',
    'fractionation_hydrogen',
    'fractionation_oxygen',
    'calc_one_wind',
    'calc_two_winds',
    
    # Coordinate utilities
    'lonlat2xy',
    'xy2lonlat',
    'wind_path',
    'catchment_nodes',
    'catchment_indices',
    
    # App functions
    'opi_calc_one_wind',
    'opi_calc_two_winds',
    'opi_fit_one_wind',
    'opi_fit_two_winds',
    'opi_plots_one_wind',
    
    # Visualization modules
    'maps',
    'plots',
    'advanced',
    'export',
    
    # I/O utilities
    'parse_run_file',
    'write_run_file',
    'grid_read',
    'get_input',
    'convert_coordinates',
    
    # Optimization
    'fmin_crs3',
    'CRS3Optimizer',
    
    # Constants
    'G', 'CPD', 'CPV', 'RD', 'RV', 'L', 'P0', 'EPSILON', 'GAMMA_D',
    'RADIUS_EARTH', 'M_PER_DEGREE', 'OMEGA',
    'TC2K', 'HR', 'SD_RES_RATIO',
    
    # Utilities
    'deuterium_excess',
    'meteoric_water_line',
    'wind_components',
    'mountain_height_number',
    'rossby_number',
    'froude_number',
    'summarize_results',
]


def print_version():
    """Print version information."""
    print(f"OPI (Orographic Precipitation and Isotopes) Python Package")
    print(f"Version: {__version__}")
    print(f"Based on MATLAB OPI 3.7 by Mark Brandon, Yale University")
