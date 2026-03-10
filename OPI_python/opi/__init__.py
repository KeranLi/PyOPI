"""
OPI (Orographic Precipitation and Isotopes) - Python Package

This package provides computational tools for the analysis of precipitation 
and isotope fractionation associated with steady atmospheric flow over 
an arbitrary three-dimensional topography.

For more information, see the documentation in README_PYTHON.md
"""

# Import main functions for easy access
from .opi_calc_one_wind import opi_calc_one_wind
from .opi_calc_two_winds import opi_calc_two_winds
from .opi_fit_one_wind import opi_fit_one_wind
from .opi_calc_two_winds import opi_calc_two_winds
from .opi_plots_one_wind import opi_plots_one_wind
from .calc_one_wind import calc_one_wind
from .base_state import base_state
from .saturated_vapor_pressure import saturated_vapor_pressure
from .coordinates import lonlat2xy, xy2lonlat
from .wind_path import wind_path
from .catchment_nodes import catchment_nodes
from .catchment_indices import catchment_indices
from .fourier_solution import fourier_solution, wind_grid
from .precipitation_grid import precipitation_grid
from .isotope_grid import isotope_grid
from .fractionation_hydrogen import fractionation_hydrogen
from .fractionation_oxygen import fractionation_oxygen
from .get_input import get_input, grid_read, estimate_mwl
from .fmin_crs3 import fmin_crs3, CRS3Optimizer
from .constants import *

__version__ = "1.0.0"
__author__ = "AI Assistant (based on original work by Mark Brandon)"
__all__ = [
    # Main functions
    'opi_calc_one_wind',
    'opi_calc_two_winds',
    'opi_fit_one_wind',
    'opi_plots_one_wind',
    'calc_one_wind',
    
    # Core utilities
    'base_state',
    'saturated_vapor_pressure',
    'lonlat2xy', 
    'xy2lonlat',
    'wind_path',
    'catchment_nodes',
    'catchment_indices',
    
    # Constants
    'G', 'CPD', 'CPV', 'RD', 'L', 'P0', 'EPSILON',
    'RADIUS_EARTH', 'M_PER_DEGREE',
    'TC2K', 'HR', 'SD_RES_RATIO'
]