# OPI Python Project Structure

## Overview

This is the organized structure of the OPI (Orographic Precipitation and Isotopes) Python package v2.0.0.

**Status**: ✅ Verified and ready for use  
**Total Modules**: 23 Python modules  
**Test Coverage**: 87+ tests

---

## Directory Tree

```
OPI_python/
│
├── opi/                          # Main Python Package (23 modules)
│   ├── __init__.py              # Package exports
│   ├── constants.py             # Physical constants
│   │
│   ├── core/                    # Core computations (9 modules)
│   │   ├── __init__.py
│   │   ├── base_state.py        # Atmospheric base state
│   │   ├── fourier_solution.py  # Fourier solution
│   │   ├── precipitation.py     # Precipitation (LTOP)
│   │   ├── isotopes.py          # Isotope calculations
│   │   ├── fractionation.py     # Fractionation factors
│   │   ├── coordinates.py       # Coordinate transforms
│   │   ├── catchment.py         # Catchment utilities
│   │   ├── lifting.py           # Orographic lifting
│   │   └── thermodynamics.py    # Thermodynamics
│   │
│   ├── models/                  # High-level API (2 modules)
│   │   ├── __init__.py
│   │   ├── one_wind.py         # OneWindModel
│   │   └── two_winds.py        # TwoWindModel
│   │
│   ├── solvers/                 # Optimization (1 module)
│   │   ├── __init__.py
│   │   └── crs3.py             # CRS3 optimizer
│   │
│   ├── io/                      # I/O utilities (2 modules)
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Load/save data
│   │   └── run_file.py         # Run file parser
│   │
│   ├── utils/                   # Utilities (3 modules)
│   │   ├── __init__.py
│   │   ├── statistics.py       # Statistics
│   │   ├── validation.py       # Validation
│   │   └── plotting.py         # Basic plotting
│   │
│   ├── visualization/           # Advanced viz (4 modules)
│   │   ├── __init__.py
│   │   ├── maps.py             # Map plots
│   │   ├── cross_sections.py   # Cross-sections
│   │   ├── analysis.py         # Analysis plots
│   │   └── summary.py          # Summary figures
│   │
│   └── parallel/                # Parallel processing (2 modules)
│       ├── __init__.py
│       ├── sweep.py            # Parameter sweeps
│       └── monte_carlo.py      # Monte Carlo
│
├── tests/                       # Test suite (8 files, 87+ tests)
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures
│   ├── core/                   # Core tests
│   ├── models/                 # Model tests
│   └── solvers/                # Solver tests
│
├── docs/                        # Sphinx documentation
│   ├── conf.py
│   ├── Makefile
│   ├── index.rst
│   ├── tutorials/              # 5 tutorials
│   └── api/                    # API reference
│
├── examples/                    # Example scripts
│   ├── simple_example.py
│   └── comprehensive_example.py
│
├── setup.py                     # Package installation
├── README.md                    # Project readme
├── verify_structure.py          # Structure verification
└── cleanup_old_files.py         # Cleanup script
```

---

## Module Details

### 1. opi/core/ - Core Computations

**Purpose**: Low-level computational functions

| File | Functionality | MATLAB Equivalent |
|------|---------------|-------------------|
| `base_state.py` | Atmospheric profiles | `baseState.m` |
| `fourier_solution.py` | Fourier solution | `fourierSolution.m` |
| `precipitation.py` | LTOP precipitation | `precipitationGrid.m` |
| `isotopes.py` | Isotope grids | `isotopeGrid.m` |
| `fractionation.py` | Fractionation | `fractionation*.m` |
| `coordinates.py` | Coordinate transforms | `lonlat2xy.m` |
| `catchment.py` | Catchments | `catchment*.m` |
| `lifting.py` | Lifting calc | `lifting.m` |
| `thermodynamics.py` | Thermodynamics | Various |

### 2. opi/models/ - High-Level Models

| File | Classes |
|------|---------|
| `one_wind.py` | `OneWindModel`, `OneWindParameters`, `OneWindResult` |
| `two_winds.py` | `TwoWindModel`, `TwoWindsParameters`, `TwoWindsResult` |

### 3. opi/solvers/ - Optimization

| File | Classes/Functions |
|------|-------------------|
| `crs3.py` | `CRS3Optimizer`, `fmin_crs3` |

### 4. opi/io/ - I/O

| File | Functions |
|------|-----------|
| `data_loader.py` | `load_topography()`, `save_results()`, `load_results()` |
| `run_file.py` | `parse_run_file()` |

### 5. opi/utils/ - Utilities

| File | Functions |
|------|-----------|
| `statistics.py` | `estimate_mwl()`, `chi_squared_test()` |
| `validation.py` | `validate_topography()`, `validate_parameters()` |
| `plotting.py` | Basic plot functions |

### 6. opi/visualization/ - Visualization

| File | Functions |
|------|-----------|
| `maps.py` | `plot_topography_contour()`, `plot_isotope_maps()` |
| `cross_sections.py` | `plot_cross_section()` |
| `analysis.py` | `plot_meteoric_water_line()` |
| `summary.py` | `create_summary_figure()` |

### 7. opi/parallel/ - Parallel Processing

| File | Classes |
|------|---------|
| `sweep.py` | `ParameterSweep`, `EnsembleRunner` |
| `monte_carlo.py` | `MonteCarloSimulator` |

---

## Usage

### Basic Import
```python
from opi import OneWindModel, OneWindParameters
```

### Core Functions
```python
from opi.core import base_state, precipitation_grid
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

---

## Verification

Run the verification script:
```bash
python verify_structure.py
```

Expected output:
```
======================================================================
OPI Package Structure Verification
======================================================================

[Core Modules] (opi/core/)
  [OK] Init: opi/core/__init__.py
  [OK] Base state: opi/core/base_state.py
  ...

[Testing Imports]
  [OK] Package 'opi' imports successfully
  [OK] opi.core imports successfully
  ...

======================================================================
SUCCESS: All checks passed! Package structure is correct.
======================================================================
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=opi --cov-report=html

# Run specific tests
pytest tests/core/test_base_state.py -v
```

---

## Documentation

```bash
cd docs/
make html
# Open _build/html/index.html
```

---

## Installation

```bash
pip install -e ".[all]"
```

---

**Last Updated**: 2024  
**Version**: 2.0.0  
**Status**: ✅ Verified
