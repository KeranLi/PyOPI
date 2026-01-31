# OPI Python Project Structure

## Overview

After refactoring, the project has a clean, modular structure with 6 main modules:

```
opi/
├── constants.py          # Physical constants (unchanged)
├── utils.py              # Utility functions (unchanged)
├── calc_one_wind.py      # Core calculation function (legacy, stable)
├── __init__.py           # Package exports
├── __main__.py           # CLI entry point
│
├── physics/              # Atmospheric physics calculations
│   ├── thermodynamics.py     # Base state, vapor pressure
│   ├── fractionation.py      # H and O isotope fractionation (merged)
│   ├── fourier.py            # FFT solution for flow
│   ├── precipitation.py      # LTOP precipitation
│   └── isotope.py            # Isotope grid calculation
│
├── io/                   # Input/output operations
│   ├── coordinates.py        # Coordinate transformations
│   └── data_loader.py        # Data loading functions
│
├── optimization/         # Optimization algorithms
│   ├── crs3.py               # CRS3 global optimizer
│   └── wind_path.py          # Wind path calculations
│
├── catchment/            # Catchment/watershed handling
│   ├── nodes.py              # Catchment node calculation
│   └── indices.py            # Catchment indices
│
└── app/                  # High-level application functions
    ├── calc_one_wind.py      # Single wind CLI
    ├── calc_two_winds.py     # Two-wind CLI
    ├── fitting.py            # Parameter fitting (merged)
    └── plotting.py           # Visualization

```

## Module Details

### physics/ - Atmospheric Physics

| File | Purpose | Original Files |
|:-----|:--------|:---------------|
| `thermodynamics.py` | Base state atmosphere, vapor pressure | `base_state.py` + `saturated_vapor_pressure.py` |
| `fractionation.py` | Isotope fractionation (H & O merged) | `fractionation_hydrogen.py` + `fractionation_oxygen.py` |
| `fourier.py` | FFT solution for flow over topography | `fourier_solution.py` |
| `precipitation.py` | LTOP precipitation calculation | `precipitation_grid.py` |
| `isotope.py` | Isotope grid calculation | `isotope_grid.py` |

### io/ - Data I/O

| File | Purpose | Original Files |
|:-----|:--------|:---------------|
| `coordinates.py` | Lon/lat to x/y conversion | `coordinates.py` |
| `data_loader.py` | Load topography and sample data | `get_input.py` |

### optimization/ - Optimization

| File | Purpose | Original Files |
|:-----|:--------|:---------------|
| `crs3.py` | CRS3 global optimization | `fmin_crs3.py` |
| `wind_path.py` | Wind path for optimization | `wind_path.py` |

### catchment/ - Catchment Handling

| File | Purpose | Original Files |
|:-----|:--------|:---------------|
| `nodes.py` | Catchment node calculation | `catchment_nodes.py` |
| `indices.py` | Catchment indices | `catchment_indices.py` |

### app/ - Applications

| File | Purpose | Original Files |
|:-----|:--------|:---------------|
| `calc_one_wind.py` | Single wind calculation CLI | `opi_calc_one_wind.py` |
| `calc_two_winds.py` | Two-wind calculation CLI | `opi_calc_two_winds.py` |
| `fitting.py` | Parameter fitting (both merged) | `opi_fit_one_wind.py` + `opi_fit_two_winds.py` |
| `plotting.py` | Plotting functions | `opi_plots_one_wind.py` |

## Import Examples

### From Package Level (Recommended)

```python
from opi import (
    base_state,                    # From physics
    fourier_solution,              # From physics
    fractionation_hydrogen,        # From physics
    lonlat2xy,                     # From io
    fmin_crs3,                     # From optimization
    catchment_nodes,               # From catchment
    opi_calc_one_wind,             # From app
    opi_fit_one_wind,              # From app
)
```

### From Specific Modules

```python
# Physics
from opi.physics import base_state, fourier_solution
from opi.physics.fractionation import fractionation_hydrogen

# IO
from opi.io import lonlat2xy, grid_read

# Optimization
from opi.optimization import fmin_crs3

# App
from opi.app import opi_calc_one_wind, opi_fit_one_wind
```

## Benefits of New Structure

1. **Clear Separation**: Each module has a single, well-defined responsibility
2. **Easier Navigation**: 23 files → 6 organized modules
3. **Reduced Coupling**: Clear dependencies between modules
4. **Better Testing**: Can test each module independently
5. **Maintained Compatibility**: All old imports still work via package-level exports

## Files Removed

The following 18 files were consolidated into the new structure:

```
base_state.py              → physics/thermodynamics.py
fourier_solution.py        → physics/fourier.py
precipitation_grid.py      → physics/precipitation.py
isotope_grid.py            → physics/isotope.py
fractionation_hydrogen.py  → physics/fractionation.py (merged)
fractionation_oxygen.py    → physics/fractionation.py (merged)
saturated_vapor_pressure.py→ physics/thermodynamics.py (merged)
coordinates.py             → io/coordinates.py
get_input.py               → io/data_loader.py
fmin_crs3.py               → optimization/crs3.py
wind_path.py               → optimization/wind_path.py
catchment_nodes.py         → catchment/nodes.py
catchment_indices.py       → catchment/indices.py
opi_calc_one_wind.py       → app/calc_one_wind.py
opi_calc_two_winds.py      → app/calc_two_winds.py
opi_fit_one_wind.py        → app/fitting.py (merged)
opi_fit_two_winds.py       → app/fitting.py (merged)
opi_plots_one_wind.py      → app/plotting.py
```

## Remaining Core Files

These 3 files remain at the package root (stable, widely used):

- `constants.py` - Used everywhere
- `utils.py` - General utilities
- `calc_one_wind.py` - Core calculation (imported by many modules)
