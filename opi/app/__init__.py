"""
Application module for OPI.

High-level application functions:
- One-wind and two-wind calculations
- Parameter fitting
- Plotting
"""

from .calc_one_wind import opi_calc_one_wind
from .calc_two_winds import opi_calc_two_winds
from .fitting import opi_fit_one_wind, opi_fit_two_winds
from .plotting import opi_plots_one_wind

__all__ = [
    'opi_calc_one_wind',
    'opi_calc_two_winds',
    'opi_fit_one_wind',
    'opi_fit_two_winds',
    'opi_plots_one_wind',
]
