API Reference
=============

This section provides detailed API documentation for all modules in the OPI package.

.. toctree::
   :maxdepth: 4

Core Modules
------------

The ``opi.core`` subpackage contains the fundamental computational routines.

.. autosummary::
   :toctree: generated
   :nosignatures:

   opi.core.base_state
   opi.core.coordinates
   opi.core.fourier_solution
   opi.core.precipitation
   opi.core.isotopes
   opi.core.fractionation
   opi.core.thermodynamics
   opi.core.lifting
   opi.core.catchment

Models
------

High-level model interfaces for running simulations.

.. autosummary::
   :toctree: generated
   :nosignatures:

   opi.models.one_wind
   opi.models.two_winds

Input/Output
------------

Utilities for loading and saving data.

.. autosummary::
   :toctree: generated
   :nosignatures:

   opi.io.data_loader
   opi.io.run_file

Solvers
-------

Optimization algorithms for parameter fitting.

.. autosummary::
   :toctree: generated
   :nosignatures:

   opi.solvers.crs3

Utilities
---------

Helper functions for visualization and analysis.

.. autosummary::
   :toctree: generated
   :nosignatures:

   opi.utils.plotting
   opi.utils.statistics
   opi.utils.validation

Package-Level Functions
-----------------------

The following functions and classes are available directly from the ``opi`` namespace:

.. currentmodule:: opi

.. autosummary::
   :nosignatures:

   ~models.OneWindModel
   ~models.TwoWindModel
   ~models.OneWindParameters
   ~models.TwoWindsParameters
   ~models.OneWindResult
   ~models.TwoWindsResult
   ~models.run_one_wind
   ~models.run_two_winds

Constants
---------

Physical constants used throughout the package:

.. currentmodule:: opi

.. list-table:: Physical Constants
   :header-rows: 1

   * - Name
     - Value
     - Description
   * - ``G``
     - 9.81 m/s²
     - Gravitational acceleration
   * - ``CPD``
     - 1004 J/(kg·K)
     - Specific heat of dry air
   * - ``CPV``
     - 1866 J/(kg·K)
     - Specific heat of water vapor
   * - ``RD``
     - 287 J/(kg·K)
     - Gas constant for dry air
   * - ``RV``
     - 461 J/(kg·K)
     - Gas constant for water vapor
   * - ``L``
     - 2.5e6 J/kg
     - Latent heat of vaporization
   * - ``GAMMA_D``
     - 9.8 K/km
     - Dry adiabatic lapse rate
   * - ``P0``
     - 101325 Pa
     - Standard sea-level pressure
   * - ``EPSILON``
     - 0.622
     - Ratio of gas constants (Rd/Rv)
   * - ``RADIUS_EARTH``
     - 6371 km
     - Earth's radius
   * - ``TC2K``
     - 273.15
     - Celsius to Kelvin offset
