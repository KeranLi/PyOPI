# OPI Python Package Structure

This document describes the organization of the OPI Python package, which is designed to mirror the MATLAB implementation while following Python best practices.

## Directory Structure

```
opi/
├── __init__.py          # Package-level imports and metadata
├── constants.py         # Physical constants and default parameters
│
├── core/                # Core computational functions (MATLAB private/)
│   ├── __init__.py
│   ├── base_state.py           # Atmospheric base state calculations
│   ├── fourier_solution.py     # Fourier solution for airflow
│   ├── precipitation.py        # Precipitation rate calculations
│   ├── isotopes.py             # Isotope fractionation and grids
│   ├── fractionation.py        # Equilibrium fractionation factors
│   ├── coordinates.py          # Coordinate transformations
│   ├── catchment.py            # Catchment/node utilities
│   ├── lifting.py              # Orographic lifting calculations
│   └── thermodynamics.py       # Thermodynamic utilities
│
├── models/              # High-level model API (MATLAB opiCalc_/opiFit_)
│   ├── __init__.py
│   ├── one_wind.py            # Single moisture source model
│   └── two_winds.py           # Two moisture source model
│
├── solvers/             # Optimization algorithms
│   ├── __init__.py
│   └── crs3.py                # CRS3 global optimizer
│
├── io/                  # Input/output utilities
│   ├── __init__.py
│   ├── data_loader.py         # Load topography and sample data
│   └── run_file.py            # Parse/write run configuration files
│
└── utils/               # Utility functions
    ├── __init__.py
    ├── statistics.py          # Statistical calculations
    ├── validation.py          # Data validation
    └── plotting.py            # Visualization utilities
```

## Module Mapping: MATLAB to Python

### MATLAB Private Functions → Python opi/core/

| MATLAB Function | Python Module | Description |
|----------------|---------------|-------------|
| `baseState.m` | `core/base_state.py` | Atmospheric base state |
| `saturatedVaporPressure.m` | `core/base_state.py` | Vapor pressure calculation |
| `fourierSolution.m` | `core/fourier_solution.py` | Fourier solution |
| `windGrid.m` | `core/fourier_solution.py` | Coordinate transformation |
| `precipitationGrid.m` | `core/precipitation.py` | Precipitation calculation |
| `isotherm.m` | `core/precipitation.py` | Isotherm height calculation |
| `isotopeGrid.m` | `core/isotopes.py` | Isotope grid calculation |
| `fractionationHydrogen.m` | `core/fractionation.py` | H isotope fractionation |
| `fractionationOxygen.m` | `core/fractionation.py` | O isotope fractionation |
| `lonlat2xy.m` | `core/coordinates.py` | Lon/lat to Cartesian |
| `xy2lonlat.m` | `core/coordinates.py` | Cartesian to lon/lat |
| `catchmentNodes.m` | `core/catchment.py` | Find catchment nodes |
| `catchmentIndices.m` | `core/catchment.py` | Extract catchment indices |
| `lifting.m` | `core/lifting.py` | Maximum lifting calculation |
| `windPath.m` | `core/lifting.py` | Wind path generation |

### MATLAB Main Programs → Python opi/models/

| MATLAB Program | Python Module | Description |
|---------------|---------------|-------------|
| `opiCalc_OneWind.m` | `models/one_wind.py` | Single-wind simulation |
| `opiCalc_TwoWinds.m` | `models/two_winds.py` | Two-wind simulation |
| `opiFit_OneWind.m` | `models/one_wind.py` | Single-wind fitting |
| `opiFit_TwoWinds.m` | `models/two_winds.py` | Two-wind fitting |
| `calc_OneWind.m` | `models/one_wind.py` | Core calculation |
| `calc_TwoWinds.m` | `models/two_winds.py` | Core calculation |

### MATLAB Optimization → Python opi/solvers/

| MATLAB Function | Python Module | Description |
|----------------|---------------|-------------|
| `fminCRS3.m` | `solvers/crs3.py` | CRS3 optimizer |

## Design Principles

### 1. Core Modules (opi/core/)

- **Pure functions**: No side effects, deterministic outputs
- **NumPy/SciPy based**: Efficient array operations
- **Well-documented**: Comprehensive docstrings with references
- **Tested**: Unit tests for each function

### 2. Model Classes (opi/models/)

- **Object-oriented**: Encapsulate state and behavior
- **Data classes**: Type-safe parameter containers
- **High-level API**: Easy-to-use interface for common tasks
- **MATLAB compatibility**: Similar function signatures

### 3. Consistent Naming

- **snake_case**: Function names (Python convention)
- **CamelCase**: Class names (Python convention)
- **Descriptive**: Clear, self-documenting names

### 4. Input/Output

- **Dictionary-based**: Flexible parameter passing
- **NumPy arrays**: Standard array format
- **Type hints**: Better IDE support and documentation

## Usage Examples

### Basic Simulation

```python
from opi import OneWindModel, OneWindParameters

# Create model
model = OneWindModel(topography={'x': x, 'y': y, 'h_grid': h})

# Set parameters
params = OneWindParameters(
    U=10.0, azimuth=270, T0=288, M=0.8,
    kappa=0, tau_c=1000, d2H0=-0.1,
    d_d2H0_d_lat=0, f_p0=1.0
)

# Run simulation
result = model.run(params)
```

### Using Core Functions Directly

```python
from opi.core import base_state, precipitation_grid

# Calculate base state
base = base_state(NM=0.01, T0=288)

# Calculate precipitation
precip = precipitation_grid(
    x, y, h_grid, U=10, azimuth=270, NM=0.01,
    f_c=1e-4, kappa=0, tau_c=1000,
    h_rho=base['h_rho'], z_bar=base['z_bar'],
    T=base['T'], gamma_env=base['gamma_env'],
    gamma_sat=base['gamma_sat'],
    gamma_ratio=base['gamma_ratio'],
    rho_s0=base['rho_s0'], h_s=base['h_s'],
    f_p0=1.0
)
```

## Migration from MATLAB

### Key Differences

1. **0-based indexing**: Python uses 0-based arrays vs MATLAB 1-based
2. **Array shapes**: (rows, cols) in Python vs (rows, cols) in MATLAB meshgrid
3. **Function outputs**: Python functions return dictionaries instead of multiple outputs
4. **Optional arguments**: Python uses keyword arguments instead of nargin checks

### Porting Tips

1. Replace MATLAB structures with Python dictionaries or dataclasses
2. Use NumPy arrays for all numerical data
3. Replace MATLAB's `fft2` with `numpy.fft.fft2`
4. Replace `griddedInterpolant` with `scipy.interpolate.RegularGridInterpolator`
5. Use `scipy.optimize.fsolve` instead of MATLAB's `fzero`

## Dependencies

- numpy: Array operations and FFT
- scipy: Optimization, interpolation, statistics
- matplotlib: Plotting and visualization
- pandas: Data I/O (optional)
- xarray: NetCDF support (optional)

## Testing

Run tests with pytest:

```bash
pytest opi/tests/
```

## Contributing

When adding new features:

1. Core algorithms go in `opi/core/`
2. High-level interfaces go in `opi/models/`
3. Add tests in `opi/tests/`
4. Update this documentation
