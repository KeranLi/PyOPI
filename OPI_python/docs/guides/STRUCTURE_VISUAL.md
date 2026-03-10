# OPI Python - Clean Project Structure

```
OPI_python/
│
├── 📁 opi/                          # Main Python Package
│   │
│   ├── 📄 __init__.py              # Package entry point - exports all public APIs
│   ├── 📄 constants.py             # Physical constants (G, RD, L, etc.)
│   │
│   ├── 📁 core/                     # Core Computational Functions (9 modules)
│   │   ├── __init__.py
│   │   ├── base_state.py           # Atmospheric base state calculations
│   │   ├── fourier_solution.py     # Fourier solution for airflow
│   │   ├── precipitation.py        # Precipitation rate (LTOP)
│   │   ├── isotopes.py             # Isotope grid calculations
│   │   ├── fractionation.py        # Fractionation factors
│   │   ├── coordinates.py          # Coordinate transformations
│   │   ├── catchment.py            # Catchment utilities
│   │   ├── lifting.py              # Orographic lifting
│   │   └── thermodynamics.py       # Thermodynamic utilities
│   │
│   ├── 📁 models/                   # High-Level Model API (2 modules)
│   │   ├── __init__.py
│   │   ├── one_wind.py            # OneWindModel, OneWindParameters
│   │   └── two_winds.py           # TwoWindModel, TwoWindsParameters
│   │
│   ├── 📁 solvers/                  # Optimization Algorithms (1 module)
│   │   ├── __init__.py
│   │   └── crs3.py                # CRS3 global optimizer
│   │
│   ├── 📁 io/                       # Input/Output (2 modules)
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Load topography (multi-format)
│   │   └── run_file.py            # Run file parsing
│   │
│   ├── 📁 utils/                    # Utility Functions (3 modules)
│   │   ├── __init__.py
│   │   ├── statistics.py          # Statistical calculations
│   │   ├── validation.py          # Data validation
│   │   └── plotting.py            # Basic plotting
│   │
│   ├── 📁 visualization/            # Advanced Visualization (4 modules)
│   │   ├── __init__.py
│   │   ├── maps.py                # 2D map plots
│   │   ├── cross_sections.py      # Cross-section plots
│   │   ├── analysis.py            # Analysis plots
│   │   └── summary.py             # Summary figures
│   │
│   └── 📁 parallel/                 # Parallel Processing (2 modules)
│       ├── __init__.py
│       ├── sweep.py               # ParameterSweep, EnsembleRunner
│       └── monte_carlo.py         # MonteCarloSimulator
│
├── 📁 tests/                        # Test Suite
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures
│   ├── 📁 core/
│   │   ├── __init__.py
│   │   ├── test_base_state.py     # Base state tests
│   │   ├── test_fourier_solution.py
│   │   └── test_fractionation.py
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   └── test_one_wind.py
│   ├── 📁 solvers/
│   │   ├── __init__.py
│   │   └── test_crs3.py
│   ├── 📁 io/
│   │   └── __init__.py
│   └── 📁 utils/
│       └── __init__.py
│
├── 📁 docs/                         # Sphinx Documentation
│   ├── conf.py
│   ├── Makefile
│   ├── index.rst
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── 📁 api/
│   │   └── modules.rst
│   ├── 📁 examples/
│   │   └── index.rst
│   └── 📁 tutorials/
│       ├── index.rst
│       ├── 01_quick_start.rst
│       ├── 02_single_wind_simulation.rst
│       ├── 03_two_winds_simulation.rst
│       ├── 04_parameter_fitting.rst
│       └── 05_working_with_real_data.rst
│
├── 📁 examples/                     # Example Scripts
│   ├── simple_example.py           # Basic usage
│   ├── comprehensive_example.py    # Full features
│   └── 📁 output/                  # Example outputs
│       └── gaussian_example/
│
├── 📁 notebooks/                    # Jupyter Notebooks
│   ├── 01_quick_start.ipynb
│   ├── 02_parameter_fitting.ipynb
│   ├── 03_data_formats.ipynb
│   ├── 04_reproduce_gaussian_example.ipynb
│   └── README.md
│
├── 📄 setup.py                      # Package Installation
├── 📄 README.md                     # Project Documentation
├── 📄 PROJECT_STRUCTURE.md          # Detailed Structure Guide
└── 📄 cleanup_old_files.py          # Cleanup Script (run once)
```

## Module Count Summary

| Directory | Modules | Purpose |
|-----------|---------|---------|
| `opi/core/` | 9 | Low-level computations |
| `opi/models/` | 2 | High-level API |
| `opi/solvers/` | 1 | Optimization |
| `opi/io/` | 2 | File I/O |
| `opi/utils/` | 3 | Utilities |
| `opi/visualization/` | 4 | Plotting |
| `opi/parallel/` | 2 | Parallel processing |
| **Total** | **23** | **Python modules** |

## Import Paths

```python
# Top-level imports (recommended)
from opi import OneWindModel, OneWindParameters
from opi import base_state, precipitation_grid
from opi import CRS3Optimizer

# Submodule imports (for specific needs)
from opi.core import base_state, fourier_solution
from opi.models import OneWindModel
from opi.visualization import create_summary_figure
from opi.parallel import ParameterSweep
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `opi/__init__.py` | ~200 | Package exports |
| `opi/constants.py` | ~30 | Physical constants |
| `opi/core/base_state.py` | ~150 | Base state calculations |
| `opi/models/one_wind.py` | ~350 | Single-wind model |
| `opi/solvers/crs3.py` | ~250 | CRS3 optimizer |

## Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/core/ -v               # Core tests only
pytest tests/models/ -v             # Model tests only
```

## Building Documentation

```bash
cd docs/
make html
# Open _build/html/index.html
```

## Installation

```bash
pip install -e ".[all]"             # Full installation
pip install -e "."                  # Basic installation
```

---

**Total Size**: ~23 Python modules, ~8 test files, ~5 tutorials
**Status**: ✅ Clean and organized
