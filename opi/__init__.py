"""
OPI (Orographic Precipitation and Isotopes) - Python Package

This package provides computational tools for the analysis of precipitation 
and isotope fractionation associated with steady atmospheric flow over 
an arbitrary three-dimensional topography.

Main modules:
- physics: Atmospheric physics calculations
- io: Data loading and coordinate transformations  
- optimization: Optimization algorithms (CRS3)
- catchment: Catchment/watershed calculations
- app: High-level application functions
"""

# Physics module
from .physics import (
    saturated_vapor_pressure,
    base_state,
    wind_grid,
    fourier_solution,
    precipitation_grid,
    isotope_grid,
    fractionation_hydrogen,
    fractionation_oxygen,
)

# IO module
from .io import (
    lonlat2xy,
    xy2lonlat,
    grid_read,
    get_input,
)

# Optimization module
from .optimization import (
    fmin_crs3,
    CRS3Optimizer,
    wind_path,
)

# Catchment module
from .catchment import (
    catchment_nodes,
    catchment_indices,
)

# App module (high-level functions)
from .app import (
    opi_calc_one_wind,
    opi_calc_two_winds,
    opi_fit_one_wind,
    opi_fit_two_winds,
    opi_plots_one_wind,
)

# Legacy core functions (still available at package level)
from .calc_one_wind import calc_one_wind
from .utils import summarize_results

# Constants
from .constants import (
    G, CPD, CPV, RD, L, P0, EPSILON,
    RADIUS_EARTH, M_PER_DEGREE,
    TC2K, HR, SD_RES_RATIO
)

__version__ = "1.0.0"
__author__ = "AI Assistant (based on original work by Mark Brandon)"

__all__ = [
    # Physics
    'base_state',
    'saturated_vapor_pressure',
    'wind_grid',
    'fourier_solution',
    'precipitation_grid',
    'isotope_grid',
    'fractionation_hydrogen',
    'fractionation_oxygen',
    # IO
    'lonlat2xy',
    'xy2lonlat',
    'grid_read',
    'get_input',
    # Optimization
    'fmin_crs3',
    'wind_path',
    # Catchment
    'catchment_nodes',
    'catchment_indices',
    # App functions
    'opi_calc_one_wind',
    'opi_calc_two_winds',
    'opi_fit_one_wind',
    'opi_fit_two_winds',
    'opi_plots_one_wind',
    # Core
    'calc_one_wind',
    # Constants
    'G', 'CPD', 'CPV', 'RD', 'L', 'P0', 'EPSILON',
    'RADIUS_EARTH', 'M_PER_DEGREE',
    'TC2K', 'HR', 'SD_RES_RATIO'
]
