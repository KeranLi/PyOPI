"""
Optimization Solvers for OPI Parameter Fitting

This module provides optimization algorithms for fitting OPI model parameters
to observational data.

Available Solvers
-----------------
CRS3Optimizer : Controlled Random Search with 3 parents
    Global optimization algorithm suitable for non-convex problems.
    This is the default optimizer used in the original MATLAB implementation.

scipy_minimize : Wrapper for SciPy optimization
    Interface to SciPy's optimization methods (L-BFGS-B, SLSQP, etc.)

Example
-------
>>> from opi.solvers import CRS3Optimizer
>>> optimizer = CRS3Optimizer(bounds=(lower_bounds, upper_bounds))
>>> result = optimizer.minimize(objective_function, x0=initial_guess)
"""

from .crs3 import CRS3Optimizer, fmin_crs3

__all__ = ['CRS3Optimizer', 'fmin_crs3']
