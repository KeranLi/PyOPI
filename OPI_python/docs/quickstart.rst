Quick Start
===========

This guide will get you up and running with OPI in just a few minutes.

Basic Example
-------------

Here's a simple example that demonstrates the core functionality of OPI:

.. code-block:: python

    import numpy as np
    import opi
    from opi.models import OneWindModel, OneWindParameters

    # Create a simple Gaussian mountain topography
    x = np.linspace(-100e3, 100e3, 200)  # x coordinates (m)
    y = np.linspace(-100e3, 100e3, 200)  # y coordinates (m)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian mountain
    h_max = 1000.0  # Maximum height (m)
    sigma = 20e3    # Width (m)
    h_grid = h_max * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    topography = {
        'x': x,
        'y': y,
        'h_grid': h_grid
    }

    # Create the model
    model = OneWindModel(topography)

    # Set up parameters
    params = OneWindParameters(
        U=10.0,           # Wind speed (m/s)
        azimuth=270.0,    # Wind direction: 270° = west to east
        T0=288.0,         # Sea-level temperature (K)
        M=0.8,            # Mountain-height number
        kappa=0.0,        # Eddy diffusivity (m²/s)
        tau_c=1000.0,     # Cloud water residence time (s)
        d2H0=-0.100,      # Base d²H (fraction = -100‰)
        d_d2H0_d_lat=0.0, # d²H latitudinal gradient
        f_p0=1.0          # No evaporative recycling
    )

    # Run the simulation
    result = model.run(params)

    # Access the results
    print(f"Precipitation rate: {result.p_grid.mean()*1000:.4f} mm/s")
    print(f"d²H range: {result.d2H_grid.min()*1000:.1f} to {result.d2H_grid.max()*1000:.1f} ‰")
    print(f"d¹⁸O range: {result.d18O_grid.min()*1000:.1f} to {result.d18O_grid.max()*1000:.1f} ‰")

Visualizing Results
-------------------

OPI includes built-in plotting utilities:

.. code-block:: python

    from opi.utils import plot_topography, plot_precipitation, plot_isotopes
    import matplotlib.pyplot as plt

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot topography
    plot_topography(topography['x'], topography['y'], topography['h_grid'], 
                    ax=axes[0, 0])
    axes[0, 0].set_title('Topography (m)')

    # Plot precipitation
    plot_precipitation(topography['x'], topography['y'], result.p_grid,
                       ax=axes[0, 1])
    axes[0, 1].set_title('Precipitation Rate (kg/m²/s)')

    # Plot d²H
    plot_isotopes(topography['x'], topography['y'], result.d2H_grid,
                  ax=axes[1, 0], label='d²H (‰)')
    axes[1, 0].set_title('d²H in Precipitation')

    # Plot d¹⁸O
    plot_isotopes(topography['x'], topography['y'], result.d18O_grid,
                  ax=axes[1, 1], label='d¹⁸O (‰)')
    axes[1, 1].set_title('d¹⁸O in Precipitation')

    plt.tight_layout()
    plt.savefig('opi_results.png', dpi=150)
    plt.show()

Using a Run File
----------------

For reproducible simulations, you can define parameters in a run file:

.. code-block:: python

    from opi.io import load_run_file

    # Load parameters from a run file
    run_config = load_run_file('my_simulation.run')
    
    # Run the simulation
    result = model.run(run_config['parameters'])

A run file (``.run``) is a simple text file with key-value pairs::

    # OPI Run File
    U = 10.0
    azimuth = 270.0
    T0 = 288.0
    M = 0.8
    tau_c = 1000.0
    d2H0 = -100.0  # in per mil

Next Steps
----------

Now that you've run your first simulation, explore more features:

* :doc:`tutorials/index` - Detailed tutorials covering all features
* :doc:`api/modules` - Complete API reference
* :doc:`examples/index` - More example scripts

Key Concepts
------------

Model Types
~~~~~~~~~~~~

OPI provides two main model classes:

* :class:`opi.models.OneWindModel` - Single wind direction, simpler and faster
* :class:`opi.models.TwoWindModel` - Two wind components for complex flow patterns

Parameters
~~~~~~~~~~~~

Key parameters you'll need to set:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Typical Values
   * - ``U``
     - Wind speed (m/s)
     - 5-30 m/s
   * - ``azimuth``
     - Wind direction (degrees)
     - 0-360°
   * - ``T0``
     - Sea-level temperature (K)
     - 280-300 K
   * - ``M``
     - Mountain-height number
     - 0.1-2.0
   * - ``tau_c``
     - Cloud water residence time (s)
     - 500-2000 s
   * - ``d2H0``
     - Base isotopic composition (fraction)
     - -0.15 to -0.05

Results
~~~~~~~

The simulation result object contains:

* ``p_grid`` - Precipitation rate (kg/m²/s)
* ``d2H_grid`` - δ²H values (fraction)
* ``d18O_grid`` - δ¹⁸O values (fraction)
* ``dexcess_grid`` - Deuterium excess (‰)
* ``P_cumulative`` - Cumulative precipitation along wind path
* Additional diagnostic variables
