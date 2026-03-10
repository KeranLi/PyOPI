"""
OPI Model API
=============

High-level model interface for running OPI simulations and optimizations.

This module provides user-friendly interfaces for:
- Single-wind and two-wind precipitation calculations
- Parameter fitting using optimization algorithms
- Batch simulations and sensitivity analysis

Classes
-------
OneWindModel : Single moisture source model
TwoWindModel : Two moisture source (mixture) model

Functions
---------
un_calculation : Run a complete OPI calculation with given parameters
fit_parameters : Fit OPI parameters to observational data

Notes
-----
These high-level functions correspond to the MATLAB main programs:
- opiCalc_OneWind.m, opiCalc_TwoWinds.m -> run_calculation
- opiFit_OneWind.m, opiFit_TwoWinds.m -> fit_parameters
"""

from .one_wind import (
    OneWindModel, 
    OneWindParameters,
    OneWindResult, 
    run_one_wind
)
from .two_winds import (
    TwoWindModel, 
    TwoWindsParameters,
    TwoWindsResult,
    run_two_winds
)

__all__ = [
    'OneWindModel',
    'OneWindParameters',
    'OneWindResult',
    'run_one_wind',
    'TwoWindModel',
    'TwoWindsParameters', 
    'TwoWindsResult',
    'run_two_winds'
]
