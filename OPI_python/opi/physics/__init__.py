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

# Thermodynamics (MATLAB converted implementation)
from .saturated_vapor_pressure import saturated_vapor_pressure
from .base_state import base_state

# Fractionation (MATLAB converted implementation)
from .fractionation_hydrogen import fractionation_hydrogen
from .fractionation_oxygen import fractionation_oxygen

# Fourier solution (MATLAB converted implementation)
from .fourier_solution import wind_grid, fourier_solution

# Precipitation and isotopes (MATLAB converted implementation)
from .precipitation_grid import precipitation_grid
from .isotope_grid import isotope_grid

# Core calculation functions (MATLAB converted implementation)
from .calc_one_wind import calc_one_wind
from .calc_two_winds import calc_two_winds

# Additional physics utilities
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
    # Precipitation and isotopes
    'precipitation_grid',
    'isotope_grid',
    # Fractionation
    'fractionation_hydrogen',
    'fractionation_oxygen',
    # Core calculations
    'calc_one_wind',
    'calc_two_winds',
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
