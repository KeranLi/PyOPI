Tutorial 2: Single Wind Simulation
==================================

This tutorial covers running simulations with a single wind direction.

Overview
--------

The :class:`opi.models.OneWindModel` is the simplest model in OPI, 
using a single prevailing wind direction.

Setting Up the Model
--------------------

.. code-block:: python

    from opi.models import OneWindModel, OneWindParameters
    
    # Create model instance
    model = OneWindModel(topography)
    
    # Define parameters
    params = OneWindParameters(
        U=10.0,          # Wind speed (m/s)
        azimuth=270.0,   # Wind direction (degrees)
        T0=288.0,        # Temperature (K)
        M=0.8,           # Mountain-height number
        tau_c=1000.0     # Cloud residence time (s)
    )

Running the Simulation
----------------------

.. code-block:: python

    result = model.run(params)
    
    # Access results
    precipitation = result.p_grid
    d2H = result.d2H_grid
    d18O = result.d18O_grid

Understanding Results
---------------------

The result object contains:

- ``p_grid``: Precipitation rate (kg/m²/s)
- ``d2H_grid``: δ²H isotope values
- ``d18O_grid``: δ¹⁸O isotope values
- ``dexcess_grid``: Deuterium excess
