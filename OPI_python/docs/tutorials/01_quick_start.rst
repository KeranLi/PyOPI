Tutorial 1: Quick Start
=======================

This tutorial covers the basics of installing and using OPI.

Installation
------------

See :doc:`../installation` for detailed installation instructions.

Your First Simulation
---------------------

1. Import the necessary modules:

.. code-block:: python

    import numpy as np
    from opi.models import OneWindModel, OneWindParameters

2. Create a simple topography (Gaussian mountain):

.. code-block:: python

    x = np.linspace(-100e3, 100e3, 200)
    y = np.linspace(-100e3, 100e3, 200)
    X, Y = np.meshgrid(x, y)
    
    h_grid = 1000 * np.exp(-(X**2 + Y**2) / (2 * (20e3)**2))
    topography = {'x': x, 'y': y, 'h_grid': h_grid}

3. Run the simulation:

.. code-block:: python

    model = OneWindModel(topography)
    params = OneWindParameters(U=10.0, azimuth=270.0, T0=288.0, M=0.8)
    result = model.run(params)

Next Steps
----------

Proceed to :doc:`02_single_wind_simulation` for more details.
