"""
Physics module for OPI.

Contains atmospheric physics calculations:
- Thermodynamics (base state, vapor pressure)
- Fourier solution for flow over topography
- Precipitation calculation (LTOP)
- Isotope fractionation and grid calculation
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
]
