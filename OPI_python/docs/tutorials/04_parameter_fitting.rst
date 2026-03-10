Tutorial 4: Parameter Fitting
=============================

This tutorial shows how to fit model parameters to observational data.

Overview
--------

OPI includes optimization tools to find the best-fit parameters 
for your specific dataset.

Setup
-----

.. code-block:: python

    from opi.models import OneWindModel
    from opi.solvers import CRS3Optimizer
    
    model = OneWindModel(topography)
    
    # Observed data
    obs_precip = ...  # Observed precipitation
    obs_d2H = ...     # Observed isotope values

Running the Optimizer
---------------------

.. code-block:: python

    optimizer = CRS3Optimizer(model, observations)
    best_params = optimizer.fit()
    
    print(f"Best wind speed: {best_params.U}")
    print(f"Best M value: {best_params.M}")

Validation
----------

Always validate fitted parameters against independent data.
