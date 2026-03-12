"""
Physics module for OPI.

Contains atmospheric physics calculations:
- Thermodynamics (base state, vapor pressure)
- Fourier solution for flow over topography
- Precipitation calculation (LTOP)
- Isotope fractionation and grid calculation
- Lifting and vertical motion
- Cloud water content
- Velocity perturbations and streamlines
"""

from .thermodynamics import saturated_vapor_pressure, base_state
from .fractionation import (
    fractionation_hydrogen, 
    fractionation_oxygen,
    fractionation_hydrogen_simple,
    fractionation_oxygen_simple
)
from .fourier import wind_grid, fourier_solution
from .precipitation import isotherm, precipitation_grid
from .isotope import isotope_grid
from .lifting import (
    calculate_vertical_velocity,
    calculate_lifting_max,
    calculate_streamlines
)
from .cloud_water import (
    calculate_cloud_water,
    calculate_relative_humidity,
    calculate_ice_water_content
)
from .velocity import (
    VelocityCalculator,
    calculate_u_prime
)

__all__ = [
    # Thermodynamics
    'saturated_vapor_pressure',
    'base_state',
    # Fourier solution
    'wind_grid',
    'fourier_solution',
    # Precipitation
    'isotherm',
    'precipitation_grid',
    # Isotopes
    'isotope_grid',
    'fractionation_hydrogen',
    'fractionation_oxygen',
    'fractionation_hydrogen_simple',
    'fractionation_oxygen_simple',
    # Lifting
    'calculate_vertical_velocity',
    'calculate_lifting_max',
    'calculate_streamlines',
    # Cloud water
    'calculate_cloud_water',
    'calculate_relative_humidity',
    'calculate_ice_water_content',
    # Velocity
    'VelocityCalculator',
    'calculate_u_prime',
]
