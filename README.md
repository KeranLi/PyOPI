# OPI Python - Orographic Precipitation and Isotopes

Python implementation of the OPI model for simulating orographic precipitation and isotope fractionation over 3D topography.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas h5py

# Run tests
python -m opi test

# Run single wind calculation
python -m opi calc-one-wind

# Run parameter fitting
python -m opi fit-one-wind
```

## Project Structure

```
opi/
├── constants.py          # Physical constants
├── utils.py              # Utility functions
├── calc_one_wind.py      # Core calculation
├── __init__.py           # Package exports
├── __main__.py           # CLI entry point
│
├── physics/              # Atmospheric physics
│   ├── thermodynamics.py     # Base state, vapor pressure
│   ├── fractionation.py      # Isotope fractionation (H & O)
│   ├── fourier.py            # FFT solution
│   ├── precipitation.py      # LTOP precipitation
│   └── isotope.py            # Isotope grids
│
├── io/                   # Input/output
│   ├── coordinates.py        # Coordinate transforms
│   └── data_loader.py        # Data loading
│
├── optimization/         # Optimization
│   ├── crs3.py               # CRS3 optimizer
│   └── wind_path.py          # Wind path
│
├── catchment/            # Catchment handling
│   ├── nodes.py              # Catchment nodes
│   └── indices.py            # Catchment indices
│
└── app/                  # Applications
    ├── calc_one_wind.py      # Single wind CLI
    ├── calc_two_winds.py     # Two-wind CLI
    ├── fitting.py            # Parameter fitting
    └── plotting.py           # Plotting
```

## Usage

### Python API

```python
import opi

# Run calculation
result = opi.opi_calc_one_wind(verbose=True)

# Access physics functions
from opi.physics import base_state, fourier_solution

# Parameter fitting
result = opi.opi_fit_one_wind(max_iterations=1000)
```

### CLI Commands

```bash
python -m opi calc-one-wind [runfile]     # Single wind calculation
python -m opi calc-two-winds [runfile]    # Two-wind calculation
python -m opi fit-one-wind [runfile]      # Parameter fitting (1 wind)
python -m opi fit-two-winds [runfile]     # Parameter fitting (2 winds)
python -m opi test                        # Run test suite
python -m opi info                        # Show package info
```

## Modules

### physics/ - Atmospheric Physics

- `base_state()` - Calculate atmospheric base state
- `fourier_solution()` - FFT solution for flow over topography
- `precipitation_grid()` - LTOP precipitation calculation
- `isotope_grid()` - Isotope distribution calculation
- `fractionation_hydrogen()` - H isotope fractionation
- `fractionation_oxygen()` - O isotope fractionation

### io/ - Data I/O

- `lonlat2xy()` / `xy2lonlat()` - Coordinate transformations
- `grid_read()` - Load topography from MAT file
- `get_input()` - Load all input data

### optimization/ - Optimization

- `fmin_crs3()` - CRS3 global optimizer
- `wind_path()` - Calculate wind paths

### catchment/ - Catchment Handling

- `catchment_nodes()` - Calculate catchment nodes
- `catchment_indices()` - Calculate catchment indices

### app/ - High-Level Applications

- `opi_calc_one_wind()` - Single wind field calculation
- `opi_calc_two_winds()` - Two wind fields calculation
- `opi_fit_one_wind()` - Parameter fitting (1 wind)
- `opi_fit_two_winds()` - Parameter fitting (2 winds)
- `opi_plots_one_wind()` - Plot results

## Dependencies

- numpy
- scipy
- matplotlib
- pandas
- h5py

## Documentation

- `PROJECT_STRUCTURE.md` - Detailed structure documentation
- `docs/` - Additional documentation

## License

See LICENSE file in parent directory.
