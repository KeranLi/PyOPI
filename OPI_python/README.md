# OPI - Orographic Precipitation and Isotopes (Python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](docs/)

A Python implementation of the **OPI (Orographic Precipitation and Isotopes)** model for analyzing precipitation and water isotope fractionation associated with steady atmospheric flow over arbitrary three-dimensional topography.

This package is a complete Python port of the original MATLAB implementation by Mark Brandon (Yale University), with additional features for modern scientific computing workflows.

## Features

- **Complete Model Implementation**: Single-wind and two-wind precipitation models
- **Isotope Calculations**: Full δ²H and δ¹⁸O fractionation modeling
- **Parallel Processing**: Parameter sweeps, ensemble runs, and Monte Carlo simulations
- **Multiple I/O Formats**: ASCII, NetCDF, GeoTIFF, MATLAB file support
- **Advanced Visualization**: Maps, cross-sections, and analysis plots
- **Optimization Tools**: CRS3 global optimizer for parameter fitting
- **Comprehensive Tests**: Full test suite with pytest

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Optional: xarray (NetCDF), rasterio (GeoTIFF), cartopy (maps), joblib (parallel)

### Basic Installation

```bash
git clone <repository-url>
cd OPI_python
pip install -e .
```

### With Optional Dependencies

```bash
# Full installation with all features
pip install -e ".[all]"

# Or specific features
pip install -e ".[io]"      # NetCDF, GeoTIFF support
pip install -e ".[viz]"     # Advanced visualization
pip install -e ".[parallel]" # Parallel processing
```

## Quick Start

```python
import numpy as np
from opi import OneWindModel, OneWindParameters

# Create topography
x = np.linspace(-50e3, 50e3, 100)
y = np.linspace(-50e3, 50e3, 100)
X, Y = np.meshgrid(x, y)
h = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20e3**2))

# Setup model
model = OneWindModel({'x': x, 'y': y, 'h_grid': h})

# Define parameters
params = OneWindParameters(
    U=10.0,        # Wind speed (m/s)
    azimuth=270.0, # Wind direction (degrees)
    T0=288.0,      # Sea-level temperature (K)
    M=0.8,         # Mountain-height number
    kappa=0.0,     # Eddy diffusivity (m²/s)
    tau_c=1000.0,  # Cloud water residence time (s)
    d2H0=-0.100,   # Base δ²H (fraction)
    d_d2H0_d_lat=0.0,  # Latitudinal gradient
    f_p0=1.0       # No evaporative recycling
)

# Run simulation
result = model.run(params)

# Access results
print(f"Mean precipitation: {result.p_grid.mean():.4f} kg/m²/s")
print(f"δ²H range: {result.d2H_grid.min()*1000:.1f}‰ to {result.d2H_grid.max()*1000:.1f}‰")
```

## Examples

### 1. Basic Simulation
See `examples/simple_example.py` for a simple Gaussian mountain simulation.

### 2. Comprehensive Workflow
See `examples/comprehensive_example.py` for advanced features including:
- Parameter sensitivity analysis
- Parameter optimization
- Advanced visualization
- Parallel processing
- I/O operations

### 3. Parameter Sweep

```python
from opi.parallel import ParameterSweep

sweep = ParameterSweep(model, n_jobs=4)
results = sweep.run_one_factor(
    'U', 
    param_values=[5, 10, 15, 20],
    base_params=params
)
```

### 4. Visualization

```python
from opi.visualization import create_summary_figure

fig = create_summary_figure(topography, result)
fig.savefig('opi_results.png', dpi=300)
```

### 5. I/O Operations

```python
from opi.io import load_topography, save_results

# Load topography from various formats
topo = load_topography('dem.tif')  # Auto-detects format

# Save results
save_results(result, 'output.nc', format='nc')  # NetCDF
save_results(result, 'output.mat', format='mat')  # MATLAB
```

## Package Structure

```
opi/
├── core/              # Core computations (MATLAB private/ equivalent)
│   ├── base_state.py
│   ├── fourier_solution.py
│   ├── precipitation.py
│   ├── isotopes.py
│   └── ...
├── models/            # High-level model API
│   ├── one_wind.py
│   └── two_winds.py
├── solvers/           # Optimization algorithms
│   └── crs3.py
├── io/                # Input/output utilities
│   ├── data_loader.py
│   └── run_file.py
├── utils/             # Utility functions
│   ├── statistics.py
│   ├── validation.py
│   └── plotting.py
├── visualization/     # Advanced visualization
│   ├── maps.py
│   ├── cross_sections.py
│   └── analysis.py
└── parallel/          # Parallel processing
    ├── sweep.py
    └── monte_carlo.py
```

## Testing

Run the full test suite with pytest:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=opi --cov-report=html

# Run specific test file
pytest tests/core/test_base_state.py -v
```

## Documentation

Build the Sphinx documentation:

```bash
cd docs/
pip install sphinx sphinx-rtd_theme
make html
```

Then open `_build/html/index.html` in your browser.

## References

This implementation is based on:

1. **Original MATLAB Code**: Mark Brandon, Yale University
2. **Smith and Barstad (2004)**: A linear theory of orographic precipitation. *J. Atmos. Sci.*, 61, 1377-1391.
3. **Durran and Klemp (1982)**: On the effects of moisture on the Brunt-Väisälä frequency. *J. Atmos. Sci.*, 39, 2152-2158.

## Citation

If you use this software in your research, please cite:

```
Brandon, M., et al. (in prep). A Linear Theory Orographic Precipitation 
    Model for Predicting the Isotopic Composition of Precipitation.

Python implementation: OPI v2.0.0, https://github.com/...
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This Python implementation maintains full compatibility with the original MATLAB algorithms while providing modern Python features for ease of use and extensibility.
