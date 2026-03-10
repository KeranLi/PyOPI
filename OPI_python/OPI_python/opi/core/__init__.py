"""
Core abstractions for OPI.

This module defines the abstract base classes and interfaces
for the OPI calculation framework.
"""

from .base import OPIBase
from .calculator import OPICalculator
from .optimizer import BaseOptimizer
from .interfaces import (
    GridCalculator,
    IsotopeCalculator,
    Solver,
    DataLoader,
    ResultExporter
)
from .exceptions import (
    OPIError,
    CalculationError,
    ValidationError,
    ConfigurationError,
    OptimizationError
)

__all__ = [
    # Base classes
    'OPIBase',
    'OPICalculator',
    'BaseOptimizer',
    # Interfaces
    'GridCalculator',
    'IsotopeCalculator',
    'Solver',
    'DataLoader',
    'ResultExporter',
    # Exceptions
    'OPIError',
    'CalculationError',
    'ValidationError',
    'ConfigurationError',
    'OptimizationError',
]
