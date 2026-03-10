<!-- 
  OPI Python - Orographic Precipitation and Isotopes
  A complete Python implementation of the OPI model with MATLAB parity
-->

<div align="center">

## PyOPI: a python implementation for *Orographic Precipitation and Isotopes over 3D Topography*🏔️
</div>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="notebooks/"><img src="https://img.shields.io/badge/Jupyter-Ready-orange.svg" alt="Jupyter"></a>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/FFT%20Solver-✓-success" alt="FFT Solver">
  <img src="https://img.shields.io/badge/LTOP%20Precipitation-✓-success" alt="LTOP Precipitation">
  <img src="https://img.shields.io/badge/Isotope%20Fractionation-✓-success" alt="Isotope Fractionation">
  <img src="https://img.shields.io/badge/CRS3%20Optimizer-✓-success" alt="CRS3 Optimizer">
  <img src="https://img.shields.io/badge/MATLAB%20Compatible-✓-success" alt="MATLAB Compatible">
</p>

---
<div align="center">

#### [📖 Documentation](docs/) • [🚀 Quick Start](#quick-start) • [📓 Tutorials](notebooks/) • [🔬 Theory](#theoretical-background)

</div>


### 🎯 Overview

OPI Python is a comprehensive implementation of the **Orographic Precipitation and Isotopes (OPI)** model, designed for atmospheric scientists, paleoclimatologists, and isotope hydrologists. The model simulates the spatial distribution of precipitation and its stable isotopic composition (δ²H, δ¹⁸O) over complex 3D topography using linear theory of orographic precipitation coupled with Rayleigh distillation.

<div align="center">
    <img width="700" alt="image" src="https://private-user-images.githubusercontent.com/66153455/543434786-cd9dfc8a-4ee9-49aa-bb7f-e5404d89b83a.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk5NTc0MjAsIm5iZiI6MTc2OTk1NzEyMCwicGF0aCI6Ii82NjE1MzQ1NS81NDM0MzQ3ODYtY2Q5ZGZjOGEtNGVlOS00OWFhLWJiN2YtZTU0MDRkODliODNhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAyMDElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMjAxVDE0NDUyMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNkZjE5ZjVlZTlmNGYzYjUxZDNkZGIyNDgzMTQzYmRiMzYyNWYzOWYyODFiZGZiMTRhMWMyZDFmNTViZGZhN2YmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.kJWpxGAjMOslH9OtD7OQ74KHala3fFaX02H3nE_zdnQ">
    </div>

#### ✨ Key Features

| Feature | Description | Status |
|:--------|:------------|:------:|
| **🌊 Atmospheric Physics** | FFT solution for linear mountain waves, LTOP microphysics, isotopic fractionation | ✅ |
| **🌀 Velocity Perturbations** | Calculation of u′ perturbations and 3D streamline tracing | ✅ |
| **⚡ Parameter Optimization** | CRS3 global optimizer for inverse modeling and Bayesian inference | ✅ |
| **🌍 Paleoclimate** | Integration with MEBM and benthic foram δ¹⁸O records | ✅ |
| **🔄 MATLAB Interop** | Read/write `.mat` files, parse legacy run files, solutions I/O | ✅ |
| **📊 Scientific Viz** | Haxby colormap, cmapscale, topographic maps, cross-sections | ✅ |
| **📓 Interactive** | Jupyter notebooks for exploratory modeling | ✅ |
| **💾 Multi-format** | MATLAB, Excel, CSV, NetCDF, GeoTIFF, JSON support | ✅ |

---

## 🧪 Theoretical Background

OPI combines **Linear Theory of Orographic Precipitation (LTOP)** with **Rayleigh distillation** to model:

1. **Orographic Lift**: Analytical FFT solution for airflow over topography
2. **Cloud Microphysics**: Box model for condensation timescales (τ_c)
3. **Isotopic Fractionation**: Equilibrium and kinetic fractionation during phase changes
4. **Moisture Tracking**: Streamline-based parcel history for isotopic evolution

The model assumes linear perturbations to a background flow, suitable for moderate topography (|h| ≪ H_scale).

---

### ⚡ Quick Start

#### Installation

```bash
git clone https://github.com/username/OPI_python.git
cd OPI_python
pip install -e .
```

Verify installation:
```bash
python -m opi test        # Run test suite
python -m opi info        # Display package information
```

---

### 🚀 Usage Methods

#### Method 1: Jupyter Notebook (Recommended for Learning)

```bash
cd OPI_python/notebooks
jupyter notebook 01_quick_start.ipynb
```

**Tutorial Series:**
| Notebook | Content |
|:---------|:--------|
| `01_quick_start.ipynb` | Basic simulation with synthetic Gaussian topography |
| `02_parameter_fitting.ipynb` | CRS3 optimization and parameter sensitivity |
| `03_data_formats.ipynb` | Loading DEMs from various sources (MATLAB, NetCDF, GeoTIFF) |

**Interactive Example:**
```python
import opi
import matplotlib.pyplot as plt

# 1. Generate synthetic topography
dem = opi.create_synthetic_dem(
    topo_type='gaussian',
    amplitude=2000,           # Mountain height [m]
    grid_size=(500e3, 500e3), # Domain size [m]
    grid_spacing=(2000, 2000) # Resolution [m]
)

# 2. Run forward simulation
result = opi.opi_calc_one_wind(verbose=True)

# 3. Visualize with Haxby colormap
fig, ax = opi.plot_topography_map(
    dem['lon'], dem['lat'], dem['hGrid'], 
    cmap=opi.haxby()
)
plt.show()
```

#### Method 2: Run Original MATLAB Example (Verification)

Reproduce the original MATLAB Gaussian Mountain example with identical parameters:

```bash
cd OPI_python/examples
python reproduce_gaussian_example.py
```

This script loads the original MATLAB topography file (`EastDirectedGaussianTopography_3km height_lat45N.mat`) and reproduces the published results with 10 m/s eastward wind, M=0.25, and T0=290K.

**Output files:**
- `gaussian_comparison_*.png` - 4-panel visualization (topography, precipitation, isotopes, cross-section)
- `gaussian_result.mat` / `gaussian_result.npz` - Simulation results in both formats
- `gaussian_summary.txt` - Parameter summary and statistics

#### Method 3: Python Script (Production Workflows)

```python
#!/usr/bin/env python
"""Production OPI simulation with explicit parameter control."""

import numpy as np
import opi
from opi.calc_one_wind import calc_one_wind

# Generate computational grid
x = np.linspace(-250000, 250000, 250)  # [m]
y = np.linspace(-250000, 250000, 250)  # [m]
X, Y = np.meshgrid(x, y)

# Define Gaussian topography
h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 50000**2))  # [m]

# Parameter vector: [U, azimuth, T0, M, kappa, tau_c, d2h0, d_d2h0_d_lat, f_p0]
beta = np.array([
    10.0,      # U: Wind speed [m/s]
    90.0,      # azimuth: Wind direction [° from N]
    290.0,     # T0: Sea-level temperature [K]
    0.25,      # M: Mountain number [-]
    0.0,       # kappa: Eddy diffusivity [m²/s]
    1000.0,    # tau_c: Condensation timescale [s]
    -5e-3,     # d2h0: Base δ²H [fraction]
    -2e-3,     # d_d2h0_d_lat: Meridional gradient [fraction/°]
    0.7        # f_p0: Residual precipitation fraction [-]
])

# Execute simulation
outputs = calc_one_wind(
    beta=beta,
    f_c=1e-4,                    # Coriolis parameter [s⁻¹]
    h_r=540,                     # Relative humidity [-]
    x=x, y=y,
    lat=np.array([45.0]),        # Latitude [°N]
    lat0=45.0,
    h_grid=h_grid,
    b_mwl_sample=np.array([9.47e-3, 8.03]),  # Meteoric water line
    # ... additional arguments
)

chi_r2, p_grid, d2h_grid, d18o_grid = outputs[0], outputs[17], outputs[23], outputs[25]

print(f"✓ Simulation complete | χ²: {chi_r2:.4f}")
print(f"  Precipitation: {p_grid.min()*1000*86400:.1f} – {p_grid.max()*1000*86400:.1f} mm/day")
```

#### Method 4: Run Files (Research Reproducibility)

**Step 1:** Create configuration file
```python
from opi.io import write_run_file

run_config = {
    'run_title': 'Alpine Gaussian Mountain Test',
    'data_path': './data',
    'topo_file': 'topography.mat',
    'sample_file': 'samples.xlsx',  # Use 'no' for forward mode
    'map_limits': [-3, 3, 42, 48],   # [lon_min, lon_max, lat_min, lat_max]
    'parameter_labels': ['U|azimuth|T0|M|kappa|tau_c|d2h0|d_d2h0_d_lat|f_p0'],
    'exponents': [0, 0, 0, 0, 0, 0, 3, 3, 0],
    'l_b': [0.1, -30, 265, 0, 0, 0, -15e-3, -5e-3, 0.5],   # Lower bounds
    'u_b': [25, 145, 295, 1.2, 1e6, 2500, 15e-3, 5e-3, 1.0], # Upper bounds
    'beta': [10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7]
}

write_run_file('alpine_simulation.run', run_config)
```

**Step 2:** Execute
```python
import opi

# Forward calculation
result = opi.opi_calc_one_wind(run_file_path='alpine_simulation.run', verbose=True)

# Parameter optimization (inverse modeling)
result = opi.opi_fit_one_wind(
    run_file_path='alpine_simulation.run', 
    verbose=True, 
    max_iterations=1000
)
```

#### Method 5: Command-Line Interface (Batch Processing)

```bash
# Forward simulation
python -m opi calc-one-wind config.run

# Parameter fitting with CRS3 optimizer
python -m opi fit-one-wind config.run --max-iter 1000

# Two-wind simulations (seasonal contrasting)
python -m opi calc-two-winds winter_summer.run
python -m opi fit-two-winds winter_summer.run
```

---

### 📂 Data Input Formats

#### Topography

| Format | Function | Usage Example |
|:-------|:---------|:--------------|
| **MATLAB** | `opi.io.grid_read()` | `x, y, h = grid_read('dem.mat')` |
| **Synthetic** | `opi.create_synthetic_dem()` | Gaussian, sinusoidal, ridge types |
| **NetCDF** | `xarray.open_dataset()` | Climate model output |
| **GeoTIFF** | `rasterio.open()` | SRTM/ASTER DEM products |
| **NumPy** | Direct array | `h_grid = np.load('dem.npy')` |

#### Isotope Samples

| Format | Required Columns | Notes |
|:-------|:-----------------|:------|
| **Excel** (.xlsx) | line, longitude, latitude, elevation, d2H, d18O, sample_type | Standard field data |
| **CSV** (.csv) | Same as above | Lightweight alternative |
| **MATLAB** (.mat) | Structured array | Legacy OPI format |

**Example sample file structure:**
```csv
line,longitude,latitude,elevation,d2H,d18O,sample_type
1,-2.0,45.0,1500,-100.0,-12.5,C
2,-1.5,45.0,1200,-95.0,-12.0,C
```

---

### ⚙️ Parameter Reference

The nine-dimensional parameter vector **β** for the OneWind model:

| # | Parameter | Symbol | Unit | Physical Meaning | Valid Range |
|:-:|:----------|:-------|:-----|:-----------------|:------------|
| 1 | **Wind speed** | U | m/s | Background flow velocity | 0.1 – 25 |
| 2 | **Azimuth** | α | ° | Wind direction (0° = N) | -30 – 145 |
| 3 | **Temperature** | T₀ | K | Sea-level air temperature | 265 – 295 |
| 4 | **Mountain number** | M | – | Nondimensional mountain height | 0 – 1.2 |
| 5 | **Eddy diffusivity** | κ | m²/s | Turbulent mixing strength | 0 – 10⁶ |
| 6 | **Condensation time** | τ_c | s | Cloud droplet growth timescale | 0 – 2500 |
| 7 | **Base δ²H** | δ²H₀ | ‰ | Isotopic composition at reference | ±15‰ |
| 8 | **δ²H gradient** | ∂δ²H/∂φ | ‰/° | Meridional isotopic gradient | ±5‰/° |
| 9 | **Precip fraction** | f_p0 | – | Residual precipitation efficiency | 0.5 – 1.0 |

*Note: Parameters 7–9 use exponents [3, 3, 0] for scaling (i.e., inputs are multiplied by 10⁻³).*

---

### 📊 Visualization

OPI provides publication-ready visualization tools:

```python
import opi
import matplotlib.pyplot as plt

# Scientific colormaps
cmap_topo = opi.haxby()        # Bathymetry/topography style
cmap_precip = opi.cmapscale(...) # Data-driven color scaling

# Standard diagnostics
fig, ax = opi.plot_topography_map(lon, lat, h_grid, cmap=cmap_topo)
fig, ax = opi.plot_precipitation_map(lon, lat, p_grid * 1000 * 86400)  # Convert to mm/day
fig, axes = opi.plot_isotope_map(lon, lat, d2h_grid, d18o_grid)

# Data-model comparison
fig, axes = opi.plot_sample_comparison(obs_d2h, obs_d18o, pred_d2h, pred_d18o)
fig, axes = opi.plot_residuals(obs_d2h, obs_d18o, pred_d2h, pred_d18o, elevation)

# Cross-sections
fig, axes = opi.plot_cross_section(x, y, h_grid, d2h_grid, line='zonal')

# Export
opi.print_figure('figure_1.pdf', dpi=600, bbox_inches='tight')
```

---

### 🔄 MATLAB Compatibility

Seamless interoperability with legacy MATLAB OPI workflows:

```python
from opi.io import load_opi_results, save_opi_results, parse_run_file

# Load existing MATLAB results
matlab_results = load_opi_results('previous_study/opiCalc_Results.mat')

# Parse legacy run files
config = parse_run_file('legacy_config.run')

# Export Python results to MATLAB format
save_opi_results('python_results.mat', {
    'chi_r2': chi_r2,
    'p_grid': p_grid,
    'd2h_grid': d2h_grid,
    'beta_optimal': beta
})
```

#### Solutions File Management

```python
from opi.io import SolutionsFileWriter, get_best_solution

# Track optimization progress
with SolutionsFileWriter() as writer:
    writer.initialize(run_path='.', run_title='Optimization', n_samples=50,
                     parameter_labels=['U', 'azimuth', 'T0', 'M', 'kappa', 
                                      'tau_c', 'd2h0', 'd_d2h0_d_lat', 'f_p0'])

    for iteration, (chi2, dof, params) in enumerate(optimizer):
        writer.write_solution(iteration, chi2, dof, params)

# Retrieve best fit
best_solution = get_best_solution('.', 'opiFit_Solutions.txt')
```

---

### 🌏 Advanced Features

#### Paleoclimate Reconstructions

Integrate with Earth system model outputs and proxy records:

```python
from opi.tools import calculate_climate_records

T0_ts, d2H0_ts, age, T0_smooth, d2H0_smooth = calculate_climate_records(
    mebm_file='MEBM_output.nc',
    benthic_file='LR04_stack.mat',
    lat=45.0,
    T0_pres=288.15,      # Present-day temperature [K]
    d2H0_pres=-100e-3,   # Present-day δ²H
    span_age=1.0         # Smoothing window [Myr]
)
```

#### 3D Streamline Tracing

```python
from opi.physics import VelocityCalculator

calc = VelocityCalculator()
calc.compute_fourier_solution(x, y, h_grid, U=10, azimuth=90, 
                             NM=0.01, f_c=1e-4)

# Trace air parcel trajectories
x_traj, y_traj, z_traj, s_dist = calc.calculate_streamline(
    x0=0, y0=0, z0=1000,  # Starting position [m]
    x=x, y=y,
    U=10, azimuth=90
)
```

---

### 🗂️ Project Structure

```
opi/
├── 📁 physics/          # Core atmospheric physics
│   ├── fft_solver.py    # Fourier solution for mountain waves
│   ├── ltop.py          # Linear theory orographic precipitation
│   └── isotopes.py      # Rayleigh distillation model
├── 📁 io/               # Data I/O and MATLAB compatibility
│   ├── matlab_io.py     # .mat file read/write
│   ├── runfiles.py      # Run file parser/generator
│   └── solutions.py     # Optimization solutions management
├── 📁 optimization/     # Parameter estimation
│   └── crs3.py          # Controlled Random Search 3 algorithm
├── 📁 viz/              # Scientific visualization
│   ├── colormaps.py     # Haxby, cmapscale
│   └── maps.py          # Geographic plotting utilities
├── 📁 notebooks/        # Tutorials
│   ├── 01_quick_start.ipynb
│   ├── 02_parameter_fitting.ipynb
│   └── 03_data_formats.ipynb
└── 📁 tests/            # Test suite
```

### 📖 Citation

If you use OPI Python in your research, please cite:

```bibtex
@software{li_2026,
  author = {Keran Li},
  title = {OPI Python: Orographic Precipitation and Isotopes},
  url = {https://github.com/KeranLi/PyOPI},
  version = {1.0.0},
  year = {2026},
}

@article{brandon_2022,
  author = {Brandon, Mark T.},
  title = {Orographic Precipitation and Isotopes (OPI) Model v3.6},
  institution = {Yale University},
  year = {2022},
  note = {Original MATLAB implementation}
}
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Bug reports and feature requests
- Code style (PEP 8, Black formatter)
- Documentation improvements
- Adding new data format support

## 📜 License

MIT License. See [LICENSE](LICENSE) file for full text.

---

<div align="center">

**Developed for the paleoclimate and isotope hydrology community**  
*Maintained by Keran Li @ [Nanjing University](https://github.com/NJU-MET)*  
*Original MATLAB version by Mark T. Brandon, Yale University*
</div>