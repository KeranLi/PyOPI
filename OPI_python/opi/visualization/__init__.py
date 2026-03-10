"""
OPI Visualization Module
=========================

This module provides visualization functions for orographic precipitation
and isotope analysis results.

Modules
-------
maps : 2D map visualizations
    Topography, precipitation, isotope, wind field, and moisture ratio plots
    
cross_sections : Cross-sectional analysis
    Extraction and plotting of cross-sections through 2D grids
    
analysis : Statistical and diagnostic plots
    Meteoric water line, residuals, and observed vs predicted plots

Example
-------
>>> from opi.visualization import plot_topography_contour, plot_isotope_maps
>>> from opi.visualization import plot_cross_section, plot_meteoric_water_line
>>>
>>> # Plot topography contours
>>> plot_topography_contour(x, y, h_grid, title='Topography')
>>>
>>> # Plot side-by-side isotope maps
>>> fig = plot_isotope_maps(x, y, d2H_grid, d18O_grid)
>>>
>>> # Create cross-section plot
>>> plot_cross_section(x, y, h_grid, p_grid, d2H_grid, 
...                    start_point=(0, 5000), end_point=(10000, 5000))
>>>
>>> # Plot meteoric water line
>>> plot_meteoric_water_line(observed, predicted)
"""

# Map visualizations
from .maps import (
    plot_topography_contour,
    plot_precipitation_contour,
    plot_isotope_maps,
    plot_wind_field,
    plot_moisture_ratio,
)

# Cross-section analysis
from .cross_sections import (
    extract_cross_section,
    plot_cross_section,
)

# Analysis plots
from .analysis import (
    plot_meteoric_water_line,
    plot_residuals,
    plot_observed_vs_predicted,
)

# Summary figure
from .summary import create_summary_figure

__all__ = [
    # Maps
    'plot_topography_contour',
    'plot_precipitation_contour',
    'plot_isotope_maps',
    'plot_wind_field',
    'plot_moisture_ratio',
    # Cross-sections
    'extract_cross_section',
    'plot_cross_section',
    # Analysis
    'plot_meteoric_water_line',
    'plot_residuals',
    'plot_observed_vs_predicted',
    # Summary
    'create_summary_figure',
]
