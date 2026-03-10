"""
OPI Core Computation Module
============================

This module contains the fundamental computational routines for the 
Orographic Precipitation and Isotopes (OPI) model.

These functions correspond to the MATLAB private functions and provide
the core physics calculations for:
- Atmospheric base state
- Fourier solutions for airflow over topography  
- Precipitation grid calculations
- Isotope fractionation and grid calculations
- Coordinate transformations
- Catchment analysis

Modules
-------
base_state : Atmospheric base state calculations
fourier_solution : Fourier solution for linearized Euler equations
precipitation : Precipitation rate and moisture calculations
isotopes : Isotope fractionation and grid calculations
fractionation : Equilibrium fractionation factors
coordinates : Geographic and wind coordinate transformations
catchment : Catchment node and index calculations
lifting : Orographic lifting calculations
thermodynamics : Thermodynamic utility functions

Notes
-----
These core functions are designed to be independent and composable,
following the structure of the original MATLAB implementation.
"""

# Base state calculations
from .base_state import base_state, saturated_vapor_pressure

# Fourier solution and coordinate transforms
from .fourier_solution import fourier_solution, wind_grid

# Precipitation calculations  
from .precipitation import precipitation_grid, isotherm

# Isotope calculations
from .isotopes import isotope_grid
from .fractionation import fractionation_hydrogen, fractionation_oxygen

# Coordinate and catchment utilities
from .coordinates import lonlat2xy, xy2lonlat
from .catchment import catchment_nodes, catchment_indices
from .lifting import lifting, wind_path
from .thermodynamics import (
    compute_lapse_rates,
    mixing_ratio_to_vapor_density,
    total_air_density
)

__all__ = [
    # Base state
    'base_state',
    'saturated_vapor_pressure',
    
    # Fourier solution
    'fourier_solution',
    'wind_grid',
    
    # Precipitation
    'precipitation_grid',
    'isotherm',
    
    # Isotopes
    'isotope_grid',
    'fractionation_hydrogen',
    'fractionation_oxygen',
    
    # Coordinates and catchment
    'lonlat2xy',
    'xy2lonlat',
    'wind_path',
    'catchment_nodes',
    'catchment_indices',
    'lifting',
    
    # Thermodynamics
    'compute_lapse_rates',
    'mixing_ratio_to_vapor_density',
    'total_air_density',
]
