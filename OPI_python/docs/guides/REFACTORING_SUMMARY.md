# OPI Python Package Refactoring Summary

## Overview

The OPI Python package has been restructured to mirror the MATLAB implementation while following Python best practices. The refactoring focuses on:

1. **Clear separation of concerns**: Core calculations, high-level models, and utilities are in separate modules
2. **MATLAB compatibility**: Similar function names and logic for easy migration
3. **Pythonic design**: Type hints, docstrings, and modern Python features
4. **Package structure**: Professional Python package layout

## New Package Structure

```
opi/
├── __init__.py              # Unified package interface
├── constants.py             # Physical constants (G, RD, L, etc.)
│
├── core/                    # Core computational functions (MATLAB private/)
│   ├── __init__.py
│   ├── base_state.py        # Atmospheric base state calculations
│   ├── fourier_solution.py  # Fourier solution for airflow
│   ├── precipitation.py     # Precipitation rate calculations
│   ├── isotopes.py          # Isotope fractionation and grids
│   ├── fractionation.py     # Equilibrium fractionation factors
│   ├── coordinates.py       # Coordinate transformations
│   ├── catchment.py         # Catchment/node utilities
│   ├── lifting.py           # Orographic lifting calculations
│   └── thermodynamics.py    # Thermodynamic utilities
│
├── models/                  # High-level model API (MATLAB opiCalc_/opiFit_)
│   ├── __init__.py
│   ├── one_wind.py         # Single moisture source model
│   └── two_winds.py        # Two moisture source model
│
├── solvers/                 # Optimization algorithms
│   ├── __init__.py
│   └── crs3.py             # CRS3 global optimizer
│
├── io/                      # Input/output utilities
│   ├── __init__.py
│   ├── data_loader.py      # Load topography and sample data
│   └── run_file.py         # Run configuration files
│
└── utils/                   # Utility functions
    ├── __init__.py
    ├── statistics.py       # Statistical calculations
    ├── validation.py       # Data validation
    └── plotting.py         # Visualization utilities
```

## Key Improvements

### 1. Core Module (opi/core/)

**MATLAB Private Functions → Python Core**

| MATLAB | Python | Status |
|--------|--------|--------|
| `baseState.m` | `core/base_state.py` | ✅ Complete |
| `saturatedVaporPressure.m` | `core/base_state.py` | ✅ Complete |
| `fourierSolution.m` | `core/fourier_solution.py` | ✅ Complete |
| `windGrid.m` | `core/fourier_solution.py` | ✅ Complete |
| `precipitationGrid.m` | `core/precipitation.py` | ✅ Complete |
| `isotherm.m` | `core/precipitation.py` | ✅ Complete |
| `isotopeGrid.m` | `core/isotopes.py` | ✅ Complete |
| `fractionationHydrogen.m` | `core/fractionation.py` | ✅ Complete |
| `fractionationOxygen.m` | `core/fractionation.py` | ✅ Complete |
| `lonlat2xy.m` | `core/coordinates.py` | ✅ Complete |
| `catchmentIndices.m` | `core/catchment.py` | ✅ Complete |
| `catchmentNodes.m` | `core/catchment.py` | ✅ Complete |
| `lifting.m` | `core/lifting.py` | ✅ Complete |

### 2. Models Module (opi/models/)

**High-Level API with Dataclasses**

```python
from opi.models import OneWindModel, OneWindParameters

# Create model
model = OneWindModel(topography={'x': x, 'y': y, 'h_grid': h})

# Define parameters with type safety
params = OneWindParameters(
    U=10.0,           # Wind speed (m/s)
    azimuth=270.0,    # Wind direction (degrees)
    T0=288.0,         # Sea-level temperature (K)
    M=0.8,            # Mountain-height number
    kappa=0.0,        # Eddy diffusivity (m²/s)
    tau_c=1000.0,     # Cloud water residence time (s)
    d2H0=-0.100,      # Base d2H (fraction)
    d_d2H0_d_lat=0.0, # Latitudinal gradient
    f_p0=1.0          # Residual precipitation
)

# Run simulation
result = model.run(params)
```

### 3. Type Safety

All core functions use type hints:

```python
def base_state(NM: float, T0: float, z_max: float = 12e3) -> dict:
    """Calculate atmospheric base state."""
    ...
```

### 4. Documentation

Each module includes:
- Module-level docstrings explaining purpose
- Function docstrings with Parameters, Returns, Notes, References
- Examples in docstrings where appropriate

### 5. Consistent API

**Dictionary Returns**: Core functions return dictionaries for clarity:

```python
# Old MATLAB style (multiple outputs)
zBar, T, gammaEnv, ... = baseState(NM, T0)

# New Python style (dictionary)
result = base_state(NM, T0)
z_bar = result['z_bar']
T = result['T']
```

## MATLAB to Python Mapping

### Function Name Conversions

| MATLAB | Python |
|--------|--------|
| `baseState` | `base_state` |
| `fourierSolution` | `fourier_solution` |
| `precipitationGrid` | `precipitation_grid` |
| `isotopeGrid` | `isotope_grid` |
| `lonlat2xy` | `lonlat2xy` |
| `catchmentNodes` | `catchment_nodes` |

### Key Differences

1. **Indexing**: MATLAB 1-based → Python 0-based
2. **Arrays**: MATLAB column-major → Python row-major (NumPy default)
3. **Structs**: MATLAB structs → Python dictionaries
4. **Multiple outputs**: MATLAB comma-separated → Python dictionary or tuple

## Usage Examples

### Simple Simulation

```python
from opi import OneWindModel, OneWindParameters
import numpy as np

# Create Gaussian mountain topography
x = np.linspace(-50e3, 50e3, 100)  # 100 km domain
y = np.linspace(-50e3, 50e3, 100)
X, Y = np.meshgrid(x, y)
h = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20e3**2))

# Setup and run model
model = OneWindModel({'x': x, 'y': y, 'h_grid': h})
params = OneWindParameters(U=10, azimuth=90, T0=288, M=0.8,
                           kappa=0, tau_c=1000, d2H0=-0.1,
                           d_d2H0_d_lat=0, f_p0=1.0)
result = model.run(params)

# Plot results
from opi import plot_topography, plot_precipitation
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_topography(x, y, h, ax=axes[0])
plot_precipitation(x, y, result.p_grid, ax=axes[1])
plt.show()
```

### Direct Core Function Usage

```python
from opi.core import base_state, fourier_solution

# Calculate atmospheric base state
base = base_state(NM=0.01, T0=288)

# Calculate Fourier solution
fourier = fourier_solution(x, y, h_grid, U=10, azimuth=270,
                           NM=0.01, f_c=1e-4, h_rho=8000)
```

## Testing the Package

```python
# Test basic imports
from opi import (
    OneWindModel, OneWindParameters,
    base_state, precipitation_grid,
    CRS3Optimizer
)

# Test core functions
result = base_state(NM=0.01, T0=288)
print(f"Water vapor scale height: {result['h_s']:.0f} m")

# Test model
model = OneWindModel({'x': x, 'y': y, 'h_grid': h})
params = OneWindParameters(
    U=10, azimuth=270, T0=288, M=0.8,
    kappa=0, tau_c=1000, d2H0=-0.1,
    d_d2H0_d_lat=0, f_p0=1.0
)
result = model.run(params)
```

## Remaining Work

### Completed
- ✅ Core module structure
- ✅ All private function implementations
- ✅ Model classes with dataclasses
- ✅ CRS3 optimizer
- ✅ Utility functions (stats, validation, plotting)
- ✅ Package-level imports
- ✅ Basic examples

### To Be Implemented
- ⬜ Full test suite (pytest)
- ⬜ NetCDF/GeoTIFF I/O support
- ⬜ Advanced plotting functions (maps, cross-sections)
- ⬜ Two-wind model fitting
- ⬜ Parallel processing support
- ⬜ Command-line interface
- ⬜ Documentation with Sphinx

## Migration Guide

### From MATLAB

1. Replace MATLAB arrays with NumPy arrays
2. Change function calls from `camelCase` to `snake_case`
3. Handle multiple outputs as dictionary keys
4. Adjust indexing (MATLAB 1→end vs Python 0→len-1)

### From Old Python Code

1. Update imports: `from opi_calc_one_wind import ...` → `from opi.models import ...`
2. Use model classes instead of standalone functions
3. Access results as attributes: `result.p_grid` instead of `p_grid`

## References

The implementation follows:
- Original MATLAB code by Mark Brandon (Yale University)
- Smith and Barstad (2004) - Linear Theory of Orographic Precipitation
- Durran and Klemp (1982) - Moisture effects on buoyancy frequency
