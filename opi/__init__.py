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
- tools: Utility tools (synthetic topography, climate records)
- viz: Visualization functions
- infrastructure: System utilities
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
    calculate_vertical_velocity,
    calculate_lifting_max,
    calculate_streamlines,
    calculate_cloud_water,
    calculate_relative_humidity,
    calculate_ice_water_content,
    VelocityCalculator,
    calculate_u_prime,
)

# IO module
from .io import (
    lonlat2xy,
    xy2lonlat,
    grid_read,
    get_input,
    load_opi_results,
    save_opi_results,
    parse_run_file,
    write_run_file,
    validate_run_data,
    SolutionsFileWriter,
    parse_solutions_file,
    get_best_solution,
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

# Tools module
from .tools import (
    gaussian_topography,
    sinusoidal_topography,
    ridge_topography,
    create_synthetic_dem,
    calculate_climate_records,
    estimate_paleoclimate_parameters,
)

# Viz module
from .viz import (
    plot_topography_map,
    plot_isotope_map,
    plot_precipitation_map,
    plot_sample_comparison,
    plot_residuals,
    plot_mwl,
    plot_dexcess,
    create_pair_plots,
    plot_prediction,
    plot_cross_section,
    haxby,
    cmapscale,
    print_figure,
)

# Infrastructure
from .infrastructure.paths import fn_persistent_path

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
    'calculate_vertical_velocity',
    'calculate_lifting_max',
    'calculate_streamlines',
    'calculate_cloud_water',
    'calculate_relative_humidity',
    'calculate_ice_water_content',
    'VelocityCalculator',
    'calculate_u_prime',
    # IO
    'lonlat2xy',
    'xy2lonlat',
    'grid_read',
    'get_input',
    'load_opi_results',
    'save_opi_results',
    'parse_run_file',
    'write_run_file',
    'validate_run_data',
    'SolutionsFileWriter',
    'parse_solutions_file',
    'get_best_solution',
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
    # Tools
    'gaussian_topography',
    'sinusoidal_topography',
    'ridge_topography',
    'create_synthetic_dem',
    'calculate_climate_records',
    'estimate_paleoclimate_parameters',
    # Viz
    'plot_topography_map',
    'plot_isotope_map',
    'plot_precipitation_map',
    'plot_sample_comparison',
    'plot_residuals',
    'plot_mwl',
    'plot_dexcess',
    'create_pair_plots',
    'plot_prediction',
    'plot_cross_section',
    'haxby',
    'cmapscale',
    'print_figure',
    # Infrastructure
    'fn_persistent_path',
    # Core
    'calc_one_wind',
    # Constants
    'G', 'CPD', 'CPV', 'RD', 'L', 'P0', 'EPSILON',
    'RADIUS_EARTH', 'M_PER_DEGREE',
    'TC2K', 'HR', 'SD_RES_RATIO'
]
