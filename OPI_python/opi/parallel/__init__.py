"""
Parallel Processing for OPI

This module provides parallel computation capabilities for OPI:
- Parameter sweep simulations
- Ensemble runs
- Sensitivity analysis
- Monte Carlo simulations

Requirements
------------
- joblib (for easy parallelization)
- tqdm (for progress bars)

Example
-------
>>> from opi.parallel import ParameterSweep, EnsembleRunner, MonteCarloSimulator
>>> from opi.models import OneWindModel
>>>
>>> model = OneWindModel(topography)
>>>
>>> # Parameter sweep
>>> sweep = ParameterSweep(model, n_jobs=4)
>>> param_grid = {'U': [5, 10, 15], 'T0': [280, 290]}
>>> results = sweep.run(param_grid, base_params={'M': 0.8, ...})
>>>
>>> # Ensemble simulation
>>> runner = EnsembleRunner(model, n_ensemble=100, n_jobs=4)
>>> ensemble = runner.run(base_params, noise_std=0.1)
>>>
>>> # Monte Carlo simulation
>>> mc = MonteCarloSimulator(model, n_samples=1000, n_jobs=4)
>>> distributions = {'U': ('normal', {'loc': 10.0, 'scale': 2.0})}
>>> results = mc.run(distributions)
>>> stats = mc.analyze_results(results)
"""

from .sweep import ParameterSweep, EnsembleRunner, SweepResult, EnsembleResult
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResult,
    MonteCarloStatistics,
    run_monte_carlo_sensitivity
)

try:
    from .distributed import DaskSimulator
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

__all__ = [
    # Sweep module
    'ParameterSweep',
    'EnsembleRunner',
    'SweepResult',
    'EnsembleResult',
    # Monte Carlo module
    'MonteCarloSimulator',
    'MonteCarloResult',
    'MonteCarloStatistics',
    'run_monte_carlo_sensitivity',
]

if HAS_DASK:
    __all__.append('DaskSimulator')
