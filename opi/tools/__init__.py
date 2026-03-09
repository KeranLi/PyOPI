"""
Tools module for OPI.

Utility tools for data generation, climate records, and processing.
"""

# Synthetic topography
from .synthetic_topography import (
    gaussian_topography,
    sinusoidal_topography,
    ridge_topography,
    create_synthetic_dem
)

# Climate records
from .climate_records import (
    load_mebm_data,
    load_benthic_foram_data,
    calculate_climate_records,
    estimate_paleoclimate_parameters
)

__all__ = [
    # Synthetic topography
    'gaussian_topography',
    'sinusoidal_topography',
    'ridge_topography',
    'create_synthetic_dem',
    # Climate records
    'load_mebm_data',
    'load_benthic_foram_data',
    'calculate_climate_records',
    'estimate_paleoclimate_parameters',
]
