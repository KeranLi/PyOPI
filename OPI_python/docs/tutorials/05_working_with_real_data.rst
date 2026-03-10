Tutorial 5: Working with Real Data
==================================

This tutorial covers loading and processing real-world datasets.

Supported Data Formats
----------------------

OPI can read:

- NetCDF files (.nc)
- HDF5 files (.h5, .hdf5)
- NumPy arrays (.npy, .npz)
- ASCII grids

Loading Topography
------------------

.. code-block:: python

    from opi.io import load_topography
    
    topography = load_topography('dem.nc')
    
    # Access the data
    x = topography['x']
    y = topography['y']
    h_grid = topography['h_grid']

Loading Observation Data
------------------------

.. code-block:: python

    from opi.io import load_observations
    
    obs = load_observations('stations.csv')

Preprocessing
-------------

Real data often requires preprocessing:

- Regridding to uniform resolution
- Handling missing values
- Unit conversions
