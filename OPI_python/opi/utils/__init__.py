"""
Utility Functions for OPI

This module provides various utility functions for:
- Statistical calculations
- Plotting and visualization
- Data validation
"""

# Statistics module
from .statistics import (
    estimate_mwl,
    calculate_residuals,
    chi_squared_test,
    calculate_mwl_residuals,
    deuterium_excess
)

# Plotting module
from .plotting import (
    plot_topography,
    plot_precipitation,
    plot_isotopes,
    plot_cross_section,
    plot_comparison,
    plot_mwl
)

# Validation module
from .validation import (
    validate_topography,
    validate_parameters,
    check_grid_consistency,
    validate_wind_parameters,
    validate_isotope_parameters
)

__all__ = [
    # Statistics
    'estimate_mwl',
    'calculate_residuals',
    'chi_squared_test',
    'calculate_mwl_residuals',
    'deuterium_excess',
    # Plotting
    'plot_topography',
    'plot_precipitation',
    'plot_isotopes',
    'plot_cross_section',
    'plot_comparison',
    'plot_mwl',
    # Validation
    'validate_topography',
    'validate_parameters',
    'check_grid_consistency',
    'validate_wind_parameters',
    'validate_isotope_parameters'
]
