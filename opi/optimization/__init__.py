"""
Optimization module for OPI.

Contains optimization algorithms:
- CRS3 (Controlled Random Search) global optimizer
- Wind path calculations for optimization
"""

from .crs3 import fmin_crs3, CRS3Optimizer
from .wind_path import wind_path

__all__ = [
    'fmin_crs3',
    'CRS3Optimizer',
    'wind_path',
]
