"""
OPI (Orographic Precipitation and Isotopes) - Python Package v2.0.0
====================================================================

A Python implementation of the OPI model for analyzing precipitation 
and water isotope fractionation associated with steady atmospheric 
flow over arbitrary three-dimensional topography.

Package Structure
-----------------
**opi.core** : Core computational functions
    Atmospheric base state, Fourier solutions, precipitation and 
    isotope calculations. These are the fundamental building blocks.

**opi.models** : High-level model interfaces  
    OneWindModel, TwoWindModel - easy-to-use classes for running
    simulations and fitting parameters.

**opi.solvers** : Optimization algorithms
    CRS3Optimizer - global optimization for parameter fitting.

**opi.io** : Input/output utilities
    Load topography (ASCII, NetCDF, GeoTIFF), sample data, and
    save results in multiple formats.

**opi.utils** : Utility functions
    Statistics, validation, basic plotting.

**opi.visualization** : Advanced visualization
    Maps, cross-sections, analysis plots with professional quality.

**opi.parallel** : Parallel processing
    Parameter sweeps, ensemble runs, Monte Carlo simulations.

Quick Start
-----------
>>> from opi import OneWindModel, OneWindParameters
>>> 
>>> # Define topography
>>> topography = {
...     'x': x_coords,
...     'y': y_coords, 
...     'h_grid': elevation_grid
... }
>>>
>>> # Create model and parameters
>>> model = OneWindModel(topography)
>>> params = OneWindParameters(
...     U=10.0,           # Wind speed (m/s)
...     azimuth=270.0,    # Wind direction (degrees from North)
...     T0=288.0,         # Sea-level temperature (K)
...     M=0.8,            # Mountain-height number
...     kappa=0.0,        # Eddy diffusivity (m²/s)
...     tau_c=1000.0,     # Cloud water residence time (s)
...     d2H0=-0.100,      # Base d2H (fraction = -100‰)
...     d_d2H0_d_lat=0.0, # d2H latitudinal gradient
...     f_p0=1.0          # No evaporative recycling
... )
>>>
>>> # Run simulation
>>> result = model.run(params)
>>>
>>> # Access results
>>> print(f"Mean precipitation: {result.p_grid.mean():.4f} kg/m²/s")
>>> print(f"d2H range: {result.d2H_grid.min()*1000:.1f}‰ to {result.d2H_grid.max()*1000:.1f}‰")

Advanced Features
-----------------
**Parallel Processing**:
>>> from opi.parallel import ParameterSweep
>>> sweep = ParameterSweep(model, n_jobs=4)
>>> results = sweep.run_one_factor('U', [5, 10, 15, 20], base_params)

**Visualization**:
>>> from opi.visualization import create_summary_figure
>>> fig = create_summary_figure(topography, result)

**I/O Operations**:
>>> from opi.io import load_topography, save_results
>>> topo = load_topography('dem.tif')  # Auto-detects format
>>> save_results(result, 'output.nc', format='nc')

References
----------
Brandon et al. (in prep) - A Linear Theory Orographic Precipitation
    Model for Predicting the Isotopic Composition of Precipitation

Smith and Barstad (2004) - A linear theory of orographic precipitation
    J. Atmos. Sci., 61, 1377-1391.

Durran and Klemp (1982) - On the effects of moisture on the 
    Brunt-Väisälä frequency. J. Atmos. Sci., 39, 2152-2158.
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "Mark Brandon, Yale University (MATLAB); AI Assistant (Python)"
__license__ = "MIT"
__docformat__ = "numpy"

# Import core functions for convenient access
from .core import (
    # Base state
    base_state,
    saturated_vapor_pressure,
    
    # Fourier solution
    fourier_solution,
    wind_grid,
    
    # Precipitation
    precipitation_grid,
    isotherm,
    
    # Isotopes
    isotope_grid,
    fractionation_hydrogen,
    fractionation_oxygen,
    
    # Coordinates
    lonlat2xy,
    xy2lonlat,
    wind_path,
    
    # Catchment
    catchment_nodes,
    catchment_indices,
    lifting,
)

# Import high-level models
from .models import (
    OneWindModel,
    TwoWindModel,
    OneWindParameters,
    TwoWindsParameters,
    OneWindResult,
    TwoWindsResult,
    run_one_wind,
    run_two_winds,
)

# Import solvers
from .solvers import (
    CRS3Optimizer,
    fmin_crs3,
)

# Import utilities
from .utils import (
    estimate_mwl,
    calculate_residuals,
    chi_squared_test,
    plot_topography,
    plot_precipitation,
    plot_isotopes,
    validate_topography,
    validate_parameters,
)

# Import visualization (if available)
try:
    from .visualization import (
        plot_topography_contour,
        plot_precipitation_contour,
        plot_isotope_maps,
        plot_cross_section,
        plot_meteoric_water_line,
        plot_residuals,
        create_summary_figure,
    )
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Import parallel processing (if available)
try:
    from .parallel import (
        ParameterSweep,
        EnsembleRunner,
        MonteCarloSimulator,
    )
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False

# Import IO functions
try:
    from .io import (
        load_topography,
        load_samples_csv,
        save_results,
        load_results,
        parse_run_file,
    )
    HAS_IO = True
except ImportError as e:
    HAS_IO = False

# Import constants
from .constants import (
    G, CPD, CPV, RD, RV, L, GAMMA_D, P0, EPSILON,
    RADIUS_EARTH, M_PER_DEGREE, OMEGA,
    TC2K, HR, SD_RES_RATIO
)

# Build __all__ based on available features
__all__ = [
    # Metadata
    '__version__',
    
    # Core functions
    'base_state',
    'saturated_vapor_pressure',
    'fourier_solution',
    'wind_grid',
    'precipitation_grid',
    'isotherm',
    'isotope_grid',
    'fractionation_hydrogen',
    'fractionation_oxygen',
    'lonlat2xy',
    'xy2lonlat',
    'wind_path',
    'catchment_nodes',
    'catchment_indices',
    'lifting',
    
    # Models
    'OneWindModel',
    'TwoWindModel',
    'OneWindParameters',
    'TwoWindsParameters',
    'OneWindResult',
    'TwoWindsResult',
    'run_one_wind',
    'run_two_winds',
    
    # Solvers
    'CRS3Optimizer',
    'fmin_crs3',
    
    # Utilities
    'estimate_mwl',
    'calculate_residuals',
    'chi_squared_test',
    'plot_topography',
    'plot_precipitation',
    'plot_isotopes',
    'validate_topography',
    'validate_parameters',
    
    # Constants
    'G', 'CPD', 'CPV', 'RD', 'RV', 'L', 'GAMMA_D', 'P0', 'EPSILON',
    'RADIUS_EARTH', 'M_PER_DEGREE', 'OMEGA',
    'TC2K', 'HR', 'SD_RES_RATIO',
]

# Add optional features to __all__
if HAS_VISUALIZATION:
    __all__.extend([
        'plot_topography_contour',
        'plot_precipitation_contour',
        'plot_isotope_maps',
        'plot_cross_section',
        'plot_meteoric_water_line',
        'plot_residuals',
        'create_summary_figure',
    ])

if HAS_PARALLEL:
    __all__.extend([
        'ParameterSweep',
        'EnsembleRunner',
        'MonteCarloSimulator',
    ])

if HAS_IO:
    __all__.extend([
        'load_topography',
        'load_samples_csv',
        'save_results',
        'load_results',
        'parse_run_file',
    ])
