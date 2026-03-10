.. OPI documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OPI Documentation
=================

Welcome to the documentation for **OPI (Orographic Precipitation and Isotopes)**, 
a Python package for analyzing precipitation and water isotope fractionation 
associated with steady atmospheric flow over arbitrary three-dimensional topography.

Overview
--------

OPI implements a linear theory orographic precipitation model that calculates:

-* **Orographic Precipitation**: Rain and snowfall resulting from terrain-induced 
  atmospheric lifting
* **Isotope Fractionation**: Distribution of stable water isotopes (δ²H, δ¹⁸O) 
  in precipitation based on Rayleigh distillation

The model extends the foundational work of Smith and Barstad (2004) with 
isotope calculations based on Rayleigh distillation theory.

Key Features
------------

* **One-Wind Model**: Single prevailing wind direction for steady-state simulations
* **Two-Winds Model**: Two wind components for more complex flow patterns  
* **Parameter Fitting**: Optimization tools to fit model parameters to observations
* **Visualization**: Built-in plotting functions for topography, precipitation, 
  and isotope distributions
* **Parallel Processing**: Support for parallel computation on multi-core systems

Quick Links
-----------

* :doc:`installation` - Install OPI and its dependencies
* :doc:`quickstart` - Get started with a simple example
* :doc:`tutorials/index` - Step-by-step tutorials
* :doc:`api/modules` - API reference documentation
* :doc:`examples/index` - Example scripts and use cases

Citation
--------

If you use OPI in your research, please cite:

.. code-block:: bibtex

   @article{brandon2026opi,
     title={A Linear Theory Orographic Precipitation Model for Predicting 
            the Isotopic Composition of Precipitation},
     author={Brandon, Mark T. and others},
     journal={Journal of ...},
     year={2026},
     note={in prep}
   }

   @article{smith2004linear,
     title={A linear theory of orographic precipitation},
     author={Smith, Ronald B. and Barstad, Ida},
     journal={Journal of the Atmospheric Sciences},
     volume={61},
     pages={1377--1391},
     year={2004}
   }

License
-------

OPI is released under the MIT License. See the LICENSE file for details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   quickstart
   tutorials/index
   api/modules
   examples/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
