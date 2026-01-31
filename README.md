# OPI Python - Orographic Precipitation and Isotopes

Complete Python implementation of the OPI model with **100% MATLAB feature parity** for simulating orographic precipitation and isotope fractionation over 3D topography.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Ready-orange.svg)](notebooks/)

## Features

- ✅ **Complete atmospheric physics** - FFT solution, LTOP precipitation, isotope fractionation
- ✅ **Velocity perturbations** - uPrime calculations and streamline tracing
- ✅ **Parameter optimization** - CRS3 global optimizer for model fitting
- ✅ **Paleoclimate support** - MEBM and benthic foram record integration
- ✅ **MATLAB compatibility** - Read/write .mat files and run files
- ✅ **Advanced visualization** - Haxby colormap, cmapscale, maps, cross-sections
- ✅ **Jupyter Notebook support** - Interactive simulations
- ✅ **Multiple data formats** - MATLAB, Excel, CSV, NetCDF, GeoTIFF, JSON
- ✅ **Solutions management** - Complete solutions file I/O
- ✅ **Command-line interface** - Full CLI for all operations

## Quick Start

### Installation

```bash
cd OPI_python
pip install -e .
```

### Run Tests

```bash
python -m opi test
python -m opi info
```

## Simulation Guide

### Method 1: Jupyter Notebook (Recommended for Learning)

```bash
cd OPI_python/notebooks
jupyter notebook 01_quick_start.ipynb
```

**Available Notebooks:**
- `01_quick_start.ipynb` - Basic simulation with synthetic data
- `02_parameter_fitting.ipynb` - Parameter optimization
- `03_data_formats.ipynb` - All supported data input formats

**Quick Jupyter Example:**

```python
import opi
import matplotlib.pyplot as plt

# 1. Create synthetic topography
dem = opi.create_synthetic_dem(
    topo_type='gaussian',
    amplitude=2000,
    grid_size=(500e3, 500e3),
    grid_spacing=(2000, 2000),
    lon0=0, lat0=45,
    output_file='topo.mat'
)

# 2. Run simulation
result = opi.opi_calc_one_wind(verbose=True)

# 3. Visualize
cmap = opi.haxby()
fig, ax = opi.plot_topography_map(dem['lon'], dem['lat'], dem['hGrid'], cmap=cmap)
plt.show()
```

### Method 2: Python Script (Recommended for Production)

```python
#!/usr/bin/env python
"""Simple OPI simulation script."""

import numpy as np
import opi
from opi.calc_one_wind import calc_one_wind

# Create grid
x = np.linspace(-250000, 250000, 250)
y = np.linspace(-250000, 250000, 250)
X, Y = np.meshgrid(x, y)

# Create topography (Gaussian mountain)
h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 50000**2))

# Define parameters [U, azimuth, T0, M, kappa, tau_c, d2h0, d_d2h0_d_lat, f_p0]
beta = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])

# Run simulation
chi_r2, nu, std_residuals, z_bar, T, gamma_env, gamma_sat, \
gamma_ratio, rho_s0, h_s, rho0, h_rho, d18o0, d_d18o0_d_lat, \
tau_f, p_grid, f_m_grid, r_h_grid, evap_d2h_grid, u_evap_d2h_grid, \
evap_d18o_grid, u_evap_d18o_grid, d2h_grid, d18o_grid, i_wet, \
d2h_pred, d18o_pred = calc_one_wind(
    beta=beta,
    f_c=1e-4,
    h_r=540,
    x=x, y=y,
    lat=np.array([45.0]),
    lat0=45.0,
    h_grid=h_grid,
    b_mwl_sample=np.array([9.47e-3, 8.03]),
    ij_catch=[[(125, 125)]],  # Simplified
    ptr_catch=[0, 1],
    sample_d2h=np.array([-100e-3]),
    sample_d18o=np.array([-12.5e-3]),
    cov=np.array([[1e-6, 0], [0, 1e-6]]),
    n_parameters_free=9,
    is_fit=False
)

print(f"Simulation complete! Chi-square: {chi_r2:.4f}")
print(f"Precipitation range: {p_grid.min()*1000*86400:.2f} - {p_grid.max()*1000*86400:.2f} mm/day")
```

### Method 3: Run File (Recommended for Research)

**Step 1: Create run file**

```python
from opi.io import write_run_file

run_data = {
    'run_title': 'My Gaussian Mountain Simulation',
    'is_parallel': False,
    'data_path': './data',
    'topo_file': 'topography.mat',
    'r_tukey': 0.25,
    'sample_file': 'samples.xlsx',  # or 'no' for forward calculation
    'cont_divide_file': None,
    'restart_file': None,
    'map_limits': [-3, 3, 42, 48],
    'section_lon0': None,
    'section_lat0': None,
    'mu': 25,
    'epsilon0': 1e-6,
    'parameter_labels': ['U|azimuth|T0|M|kappa|tau_c|d2h0|d_d2h0_d_lat|f_p0'],
    'exponents': [0, 0, 0, 0, 0, 0, 3, 3, 0],
    'l_b': [0.1, -30, 265, 0, 0, 0, -15e-3, -5e-3, 0.5],
    'u_b': [25, 145, 295, 1.2, 1e6, 2500, 15e-3, 5e-3, 1.0],
    'beta': [10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7]
}

write_run_file('my_run.run', run_data)
```

**Step 2: Execute simulation**

```python
import opi

# Forward calculation
result = opi.opi_calc_one_wind(run_file_path='my_run.run', verbose=True)

# Parameter fitting
result = opi.opi_fit_one_wind(run_file_path='my_run.run', verbose=True, max_iterations=1000)
```

### Method 4: CLI (Command Line)

```bash
# Forward calculation
python -m opi calc-one-wind my_run.run

# Parameter fitting
python -m opi fit-one-wind my_run.run

# Two-wind simulation
python -m opi calc-two-winds my_run_two_winds.run
python -m opi fit-two-winds my_run_two_winds.run
```

## Data Input Formats

OPI supports multiple data formats:

### Topography Data

| Format | Function | Example |
|:-------|:---------|:--------|
| MATLAB .mat | `opi.io.grid_read()` | `x, y, h = grid_read('topo.mat')` |
| Synthetic | `opi.create_synthetic_dem()` | `dem = create_synthetic_dem(...)` |
| NumPy array | Direct | `h_grid = np.array(...)` |
| NetCDF | `xarray.open_dataset()` | `ds = xr.open_dataset('topo.nc')` |
| GeoTIFF | `rasterio.open()` | `src = rasterio.open('topo.tif')` |

### Sample Data (Isotope Measurements)

| Format | Function | Required Columns |
|:-------|:---------|:-----------------|
| Excel .xlsx | `pd.read_excel()` | line, longitude, latitude, elevation, d2H, d18O, sample_type |
| CSV | `pd.read_csv()` | Same as Excel |
| NumPy | Direct | Arrays of sample coordinates and isotope values |

**Example Excel/CSV format:**

```
line,longitude,latitude,elevation,d2H,d18O,d_excess,sample_type
1,-2.0,45.0,1500,-100,-12.5,10,C
2,-1.5,45.0,1200,-95,-12.0,10,C
...
```

### Example: Loading Different Data Formats

```python
import opi
import pandas as pd
import numpy as np

# 1. From MATLAB file
from opi.io import grid_read
x, y, h_grid = grid_read('my_topography.mat')

# 2. From Excel/CSV
samples = pd.read_excel('my_samples.xlsx')
# or
samples = pd.read_csv('my_samples.csv')

# 3. From NetCDF (climate data)
import xarray as xr
ds = xr.open_dataset('climate_data.nc')
h_grid = ds['topography'].values

# 4. From GeoTIFF (GIS)
import rasterio
with rasterio.open('dem.tif') as src:
    h_grid = src.read(1)

# 5. Create synthetic
dem = opi.create_synthetic_dem(
    topo_type='gaussian',      # or 'sinusoidal_east', 'sinusoidal_north', 'ridge'
    amplitude=2000,
    grid_size=(500e3, 500e3),
    grid_spacing=(2000, 2000)
)
x, y, h_grid = dem['x'], dem['y'], dem['hGrid']
```

## Parameter Reference

The 9 parameters for OneWind model:

| # | Parameter | Symbol | Unit | Description | Typical Range |
|:-:|:----------|:-------|:-----|:------------|:--------------|
| 1 | Wind speed | U | m/s | Horizontal wind speed | 0.1 - 25 |
| 2 | Azimuth | azimuth | ° | Wind direction (from North) | -30 - 145 |
| 3 | Temperature | T0 | K | Sea-level temperature | 265 - 295 |
| 4 | Mountain number | M | - | Mountain height number | 0 - 1.2 |
| 5 | Eddy diffusivity | kappa | m²/s | Turbulent diffusion | 0 - 1e6 |
| 6 | Condensation time | tau_c | s | Cloud condensation timescale | 0 - 2500 |
| 7 | Base d2H | d2h0 | fraction | Isotopic composition at base | ±15e-3 |
| 8 | d2H gradient | d_d2h0_d_lat | fraction/° | Latitudinal gradient | ±5e-3 |
| 9 | Precip fraction | f_p0 | - | Residual precipitation | 0.5 - 1.0 |

## Visualization

```python
import opi
import matplotlib.pyplot as plt

# Haxby colormap (oceanographic style)
cmap = opi.haxby()

# Basic maps
fig, ax = opi.plot_topography_map(lon, lat, h_grid, cmap=cmap)
fig, ax = opi.plot_precipitation_map(lon, lat, p_grid)
fig, axes = opi.plot_isotope_map(lon, lat, d2h_grid, d18o_grid)

# Standard plots
fig, axes = opi.plot_sample_comparison(obs_d2h, obs_d18o, pred_d2h, pred_d18o)
fig, axes = opi.plot_residuals(obs_d2h, obs_d18o, pred_d2h, pred_d18o, elevation)
fig, ax = opi.plot_mwl(d18o, d2h)

# Advanced plots
fig, axes = opi.create_pair_plots({'d2H': d2h, 'd18O': d18o, 'elev': elev})
fig, axes = opi.plot_cross_section(x, y, h_grid, d2h_grid)

# Data-driven colormap
cmap_scaled, ticks, labels = opi.cmapscale(
    z=precip_grid,
    cmap0=plt.cm.coolwarm(np.linspace(0, 1, 256)),
    factor=0.8,      # Contrast (0-1)
    z0=0,            # Center value
    n_ticks=11,
    n_round=0
)

# Export figure
opi.print_figure('output.pdf', dpi=600)
```

## MATLAB Compatibility

The Python version provides **100% compatibility** with MATLAB OPI:

```python
from opi.io import load_opi_results, save_opi_results, parse_run_file

# Load MATLAB results
results = load_opi_results('opiCalc_Results.mat')

# Parse MATLAB run file
run_data = parse_run_file('my_run.run')

# Save results in MATLAB format
save_opi_results('python_results.mat', results)
```

### Solutions File I/O

```python
from opi.io import SolutionsFileWriter, parse_solutions_file, get_best_solution

# Write solutions during optimization
with SolutionsFileWriter() as writer:
    writer.initialize(
        run_path='.',
        run_title='My Optimization',
        n_samples=10,
        parameter_labels=['U', 'azimuth', 'T0', 'M', 'kappa', 
                         'tau_c', 'd2h0', 'd_d2h0_d_lat', 'f_p0'],
        exponents=[0, 0, 0, 0, 0, 0, 3, 3, 0],
        lb=[0.1, -30, 265, 0, 0, 0, -15e-3, -5e-3, 0.5],
        ub=[25, 145, 295, 1.2, 1e6, 2500, 15e-3, 5e-3, 1.0]
    )
    for i, (chi_r2, nu, beta) in enumerate(solutions):
        writer.write_solution(i+1, chi_r2, nu, beta)

# Parse solutions file
results = parse_solutions_file('.', 'opiFit_Solutions.txt')
best = get_best_solution('.', 'opiFit_Solutions.txt')
```

## Advanced Features

### Paleoclimate Records

```python
from opi.tools import calculate_climate_records

T0_record, d2H0_record, age_record, T0_smooth, d2H0_smooth = \
    calculate_climate_records(
        mebm_file='MEBM_vary_OLR.mat',
        benthic_file='Cramer2011BenthicForamClimate.mat',
        lat=45.0,
        T0_pres=288.15,
        d2H0_pres=-100e-3,
        d_d2H0_d_d18O0=8.0,
        d_d2H0_d_T0=4.5e-3,
        span_age=1.0
    )
```

### Velocity Calculations

```python
from opi.physics import VelocityCalculator

calc = opi.VelocityCalculator()
calc.compute_fourier_solution(x, y, h_grid, U=10, azimuth=90, 
                               NM=0.01, f_c=1e-4, h_rho=8000)

# Calculate velocity perturbation
u_prime = calc.calculate_u_prime(z_bar=1000, U=10, f_c=1e-4, h_rho=8000)

# Calculate streamline
x_l, y_l, z_l, s_l = calc.calculate_streamline(
    x_l0=0, y_l0=0, z_l0=1000, azimuth=90, x=x, y=y,
    U=10, f_c=1e-4, h_rho=8000
)
```

## Project Structure

```
opi/
├── physics/          # Atmospheric physics (8 modules)
├── io/               # Data I/O (5 modules)
├── optimization/     # Optimization (2 modules)
├── catchment/        # Catchment (2 modules)
├── app/              # Applications (4 modules)
├── tools/            # Tools (2 modules)
├── viz/              # Visualization (5 modules)
├── infrastructure/   # System utilities
└── notebooks/        # Jupyter notebooks (3 tutorials)
```

## Dependencies

- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.4
- pandas >= 1.3
- h5py >= 3.0
- jupyter (optional, for notebooks)
- xarray (optional, for NetCDF)
- rasterio (optional, for GeoTIFF)

## Documentation

- `README.md` - This file (overview and quick start)
- `PROJECT_STRUCTURE.md` - Detailed project structure
- `docs/MATLAB_PYTHON_COMPARISON.md` - Complete MATLAB vs Python comparison
- `notebooks/` - Interactive tutorials

## Citation

Based on the original MATLAB OPI model by Mark Brandon, Yale University.

```
Brandon, M.T. (2022) Orographic Precipitation and Isotopes (OPI) Model v3.6
```

## License

MIT License - See LICENSE file in parent directory.

## Support

For issues and questions:
1. Check the Jupyter notebooks in `notebooks/`
2. Review `docs/MATLAB_PYTHON_COMPARISON.md` for migration help
3. Run `python -m opi info` for package information
