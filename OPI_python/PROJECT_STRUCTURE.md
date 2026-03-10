# OPI Python Project Structure

This document describes the organized structure of the OPI Python package.

## Directory Overview

```
OPI_python/
├── opi/                      # Main Python package
│   ├── core/                 # Core computational functions
│   ├── models/               # High-level model interfaces
│   ├── solvers/              # Optimization algorithms
│   ├── io/                   # Input/output utilities
│   ├── utils/                # Utility functions
│   ├── visualization/        # Advanced visualization
│   ├── parallel/             # Parallel processing
│   ├── constants.py          # Physical constants
│   └── __init__.py           # Package initialization
│
├── tests/                    # Test suite
├── docs/                     # Sphinx documentation
├── examples/                 # Example scripts
├── notebooks/                # Jupyter notebooks
├── setup.py                  # Package installation
└── README.md                 # Project readme
```

## Module Details

### 1. opi/core/ - Core Computations

**Purpose**: Low-level computational functions (equivalent to MATLAB `private/`)

| File | Description | MATLAB Equivalent |
|------|-------------|-------------------|
| `base_state.py` | Atmospheric base state calculations | `baseState.m` |
| `fourier_solution.py` | Fourier solution for airflow | `fourierSolution.m` |
| `precipitation.py` | Precipitation rate calculations | `precipitationGrid.m` |
| `isotopes.py` | Isotope grid calculations | `isotopeGrid.m` |
| `fractionation.py` | Fractionation factors | `fractionation*.m` |
| `coordinates.py` | Coordinate transformations | `lonlat2xy.m`, `xy2lonlat.m` |
| `catchment.py` | Catchment utilities | `catchment*.m` |
| `lifting.py` | Orographic lifting | `lifting.m` |
| `thermodynamics.py` | Thermodynamic utilities | Various |

**Exports**:
```python
from opi.core import base_state, fourier_solution, precipitation_grid
from opi.core import isotope_grid, fractionation_hydrogen, lonlat2xy
```

### 2. opi/models/ - High-Level Models

**Purpose**: User-friendly model interfaces

| File | Description |
|------|-------------|
| `one_wind.py` | Single moisture source model |
| `two_winds.py` | Two moisture source model |

**Classes**:
- `OneWindModel` / `TwoWindModel` - Main model classes
- `OneWindParameters` / `TwoWindsParameters` - Parameter dataclasses
- `OneWindResult` / `TwoWindsResult` - Result dataclasses

**Exports**:
```python
from opi.models import OneWindModel, OneWindParameters, run_one_wind
from opi.models import TwoWindModel, TwoWindsParameters, run_two_winds
```

### 3. opi/solvers/ - Optimization

**Purpose**: Parameter fitting algorithms

| File | Description |
|------|-------------|
| `crs3.py` | CRS3 global optimizer |

**Exports**:
```python
from opi.solvers import CRS3Optimizer, fmin_crs3
```

### 4. opi/io/ - Input/Output

**Purpose**: File loading and saving

| File | Description | Formats |
|------|-------------|---------|
| `data_loader.py` | Load/save data | NPZ, NetCDF, GeoTIFF, MATLAB |
| `run_file.py` | Run file parsing | Custom run files |

**Exports**:
```python
from opi.io import load_topography, save_results, load_results
from opi.io import load_samples_csv, parse_run_file
```

### 5. opi/utils/ - Utilities

**Purpose**: Helper functions

| File | Description |
|------|-------------|
| `statistics.py` | Statistical calculations |
| `validation.py` | Data validation |
| `plotting.py` | Basic plotting |

**Exports**:
```python
from opi.utils import estimate_mwl, calculate_residuals, chi_squared_test
from opi.utils import validate_topography, validate_parameters
```

### 6. opi/visualization/ - Advanced Visualization

**Purpose**: Publication-quality figures

| File | Description |
|------|-------------|
| `maps.py` | 2D map visualizations |
| `cross_sections.py` | Cross-sectional plots |
| `analysis.py` | Statistical plots |
| `summary.py` | Summary figures |

**Exports**:
```python
from opi.visualization import plot_topography_contour, plot_isotope_maps
from opi.visualization import plot_cross_section, create_summary_figure
from opi.visualization import plot_meteoric_water_line, plot_residuals
```

### 7. opi/parallel/ - Parallel Processing

**Purpose**: High-performance computing

| File | Description |
|------|-------------|
| `sweep.py` | Parameter sweeps and ensemble runs |
| `monte_carlo.py` | Monte Carlo simulations |

**Exports**:
```python
from opi.parallel import ParameterSweep, EnsembleRunner
from opi.parallel import MonteCarloSimulator
```

## File Status

### ✅ Active Files (New Structure)

These files are part of the new organized structure:

**Core (9 files)**:
- `opi/core/base_state.py`
- `opi/core/fourier_solution.py`
- `opi/core/precipitation.py`
- `opi/core/isotopes.py`
- `opi/core/fractionation.py`
- `opi/core/coordinates.py`
- `opi/core/catchment.py`
- `opi/core/lifting.py`
- `opi/core/thermodynamics.py`

**Models (2 files)**:
- `opi/models/one_wind.py`
- `opi/models/two_winds.py`

**Solvers (1 file)**:
- `opi/solvers/crs3.py`

**IO (2 files)**:
- `opi/io/data_loader.py`
- `opi/io/run_file.py`

**Utils (3 files)**:
- `opi/utils/statistics.py`
- `opi/utils/validation.py`
- `opi/utils/plotting.py`

**Visualization (4 files)**:
- `opi/visualization/maps.py`
- `opi/visualization/cross_sections.py`
- `opi/visualization/analysis.py`
- `opi/visualization/summary.py`

**Parallel (2 files)**:
- `opi/parallel/sweep.py`
- `opi/parallel/monte_carlo.py`

### ⚠️ Deprecated Files (To Be Removed)

These old files exist at the `opi/` root level but should be removed after migration:

```
opi/base_state.py                   → Use opi/core/base_state.py
opi/calc_one_wind.py                → Use opi/models/one_wind.py
opi/calc_two_winds.py               → Use opi/models/two_winds.py
opi/catchment_indices.py            → Use opi/core/catchment.py
opi/catchment_nodes.py              → Use opi/core/catchment.py
opi/coordinates.py                  → Use opi/core/coordinates.py
opi/fmin_crs3.py                    → Use opi/solvers/crs3.py
opi/fourier_solution.py             → Use opi/core/fourier_solution.py
opi/fractionation_hydrogen.py       → Use opi/core/fractionation.py
opi/fractionation_oxygen.py         → Use opi/core/fractionation.py
opi/get_input.py                    → Use opi/io/data_loader.py
opi/isotope_grid.py                 → Use opi/core/isotopes.py
opi/opi_calc_one_wind.py            → Use opi/models/one_wind.py
opi/opi_calc_two_winds.py           → Use opi/models/two_winds.py
opi/opi_fit_one_wind.py             → Use opi/models/one_wind.py
opi/opi_fit_two_winds.py            → Use opi/models/two_winds.py
opi/opi_maps_one_wind.py            → Use opi/visualization/
opi/opi_maps_two_winds.py           → Use opi/visualization/
opi/opi_plots_one_wind.py           → Use opi/visualization/
opi/precipitation_grid.py           → Use opi/core/precipitation.py
opi/run_opi_simulation.py           → Use opi/models/
opi/saturated_vapor_pressure.py     → Use opi/core/base_state.py
opi/utils.py                        → Use opi/utils/
opi/wind_path.py                    → Use opi/core/lifting.py
```

## Import Hierarchy

```
opi/                                # Public API
├── __init__.py                     # Exports all public functions
├── constants.py                    # Physical constants
│
├── core/                           # Internal - core computations
│   └── __init__.py                 # Exports core functions
│
├── models/                         # Public - main user interface
│   └── __init__.py                 # Exports model classes
│
├── solvers/                        # Public - optimization
│   └── __init__.py                 # Exports solvers
│
├── io/                             # Public - I/O operations
│   └── __init__.py                 # Exports I/O functions
│
├── utils/                          # Public - utilities
│   └── __init__.py                 # Exports utilities
│
├── visualization/                  # Public - plotting (optional)
│   └── __init__.py                 # Exports visualization
│
└── parallel/                       # Public - parallel (optional)
    └── __init__.py                 # Exports parallel tools
```

## Usage Examples

### Basic Import
```python
import opi
from opi import OneWindModel, OneWindParameters
```

### Core Functions
```python
from opi.core import base_state, precipitation_grid
base = base_state(NM=0.01, T0=288)
```

### Models
```python
from opi.models import OneWindModel
model = OneWindModel(topography)
result = model.run(params)
```

### I/O
```python
from opi.io import load_topography, save_results
topo = load_topography('dem.nc')
save_results(result, 'output.nc')
```

### Visualization
```python
from opi.visualization import create_summary_figure
fig = create_summary_figure(topo, result)
```

### Parallel
```python
from opi.parallel import ParameterSweep
sweep = ParameterSweep(model, n_jobs=4)
```

## Migration Guide

### From Old Files

If you were using the old flat structure:

```python
# OLD (deprecated)
from base_state import base_state
from precipitation_grid import precipitation_grid

# NEW (correct)
from opi.core import base_state, precipitation_grid
```

### From MATLAB

```python
# MATLAB
# [zBar, T, ...] = baseState(NM, T0);

# Python
from opi.core import base_state
result = base_state(NM=0.01, T0=288)
z_bar = result['z_bar']
T = result['T']
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/core/ -v
pytest tests/models/ -v
```

## Documentation

```bash
# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

---

**Last Updated**: 2024
**Version**: 2.0.0
