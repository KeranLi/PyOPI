"""
OPI Data Models

This module contains all data classes for OPI, providing type-safe
and structured representations of parameters, results, and configurations.
"""

from .domain import Grid, Topography, CoordinateSystem
from .parameters import (
    OneWindParameters,
    TwoWindParameters,
    WindField,
    AtmosphereState,
    IsotopeParameters,
    MicrophysicsParameters,
    Sample,
    SampleCollection
)
from .results import (
    OneWindResults,
    TwoWindResults,
    PrecipitationGrid,
    IsotopeGrid,
    FourierSolution
)
from .config import OPIConfig, SolverConfig, OptimizerConfig

__all__ = [
    # Domain
    'Grid',
    'Topography', 
    'CoordinateSystem',
    # Parameters
    'OneWindParameters',
    'TwoWindParameters',
    'WindField',
    'AtmosphereState',
    # Results
    'OneWindResults',
    'TwoWindResults',
    'PrecipitationGrid',
    'IsotopeGrid',
    'FourierSolution',
    # Config
    'OPIConfig',
    'SolverConfig',
    'OptimizerConfig',
]
