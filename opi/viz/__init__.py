"""
Visualization module for OPI.

Provides plotting, mapping, and export functions for OPI results.
"""

# Maps
from .maps import (
    plot_topography_map,
    plot_precipitation_map,
    plot_isotope_map,
    plot_result_maps
)

# Standard plots
from .plots import (
    plot_sample_comparison,
    plot_residuals,
    plot_mwl,
    plot_dexcess
)

# Advanced plots
from .advanced import (
    create_pair_plots,
    plot_prediction,
    plot_cross_section
)

# Colormaps
from .colormaps import (
    haxby,
    cmapscale,
    cool_warm,
    create_centered_colormap,
    OPI_COLORMAPS
)

# Export
from .export import (
    print_figure,
    save_figure_set
)

__all__ = [
    # Maps
    'plot_topography_map',
    'plot_precipitation_map',
    'plot_isotope_map',
    'plot_result_maps',
    # Plots
    'plot_sample_comparison',
    'plot_residuals',
    'plot_mwl',
    'plot_dexcess',
    # Advanced
    'create_pair_plots',
    'plot_prediction',
    'plot_cross_section',
    # Colormaps
    'haxby',
    'cmapscale',
    'cool_warm',
    'create_centered_colormap',
    'OPI_COLORMAPS',
    # Export
    'print_figure',
    'save_figure_set',
]
