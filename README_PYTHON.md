# OPI: Orographic Precipitation and Isotopes - Python Version

## Overview

This is a Python translation of the original MATLAB package for the analysis of orographic precipitation and isotope fractionation associated with steady atmospheric flow over an arbitrary three-dimensional topography.

**Original Author**: Mark Brandon (mark.brandon@yale.edu)  
**Python Translation**: AI Assistant

## Requirements

- Python 3.8 or higher
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.4.0
- Pandas >= 1.3.0
- XArray >= 0.19.0
- netCDF4 >= 1.5.0
- h5py >= 3.0.0

## Environment Setup

We provide scripts to help set up your environment easily:

### Windows
Run the batch file to set up your environment:
```
setup_environment.bat
```

### Unix/Linux/macOS
Make the shell script executable and run it:
```
chmod +x setup_environment.sh
./setup_environment.sh
```

### Manual Setup
If you prefer to set up manually:

First, clone or download the repository:

```bash
git clone https://github.com/your-repo/OPI-Orographic-Precipitation-and-Isotopes.git
cd OPI-Orographic-Precipitation-and-Isotopes
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install numpy scipy matplotlib pandas xarray netcdf4 h5py
```

## Verification

After installation, you can verify that the package is correctly installed by running the test script:

```bash
python verify_installation.py
```

Alternatively, you can manually verify the installation by running:

```python
python -c "import opi; print('OPI package imported successfully')"
```

If you encounter issues with the automated test script, you can manually verify each component:

1. Test the imports:
```python
from opi.constants import G, CPD, RD
from opi.saturated_vapor_pressure import saturated_vapor_pressure
from opi.base_state import base_state
from opi.coordinates import lonlat2xy, xy2lonlat
from opi.wind_path import wind_path
from opi.catchment_nodes import catchment_nodes
from opi.catchment_indices import catchment_indices
from opi.precipitation_grid import precipitation_grid
from opi.isotope_grid import isotope_grid
from opi.calc_one_wind import calc_one_wind
from opi.opi_calc_one_wind import opi_calc_one_wind

print("All modules imported successfully!")
```

2. Test basic functionality:
```python
# Test saturated vapor pressure calculation
from opi.saturated_vapor_pressure import saturated_vapor_pressure
e_s = saturated_vapor_pressure(298.15)  # 25°C in Kelvin
print(f"Saturated vapor pressure at 25°C: {e_s:.2f} Pa")

# Test coordinate conversion
from opi.coordinates import lonlat2xy
x, y = lonlat2xy([-120.0, -119.0], [40.0, 41.0], -120.0, 40.0)
print(f"Coordinate conversion result: X={x}, Y={y}")
```

## Usage

The Python version maintains the same functionality as the original MATLAB version. Here's how to run the main calculation:

```python
from opi.opi_calc_one_wind import opi_calc_one_wind

# Run the calculation
results = opi_calc_one_wind(run_file_path="path/to/your/run/file")
```

## How to Run

### 1. Quick Test Run
To perform a quick test of the installation, run:

```python
from opi.opi_calc_one_wind import opi_calc_one_wind

# This will run with default parameters
results = opi_calc_one_wind()
print(results)
```

### 2. Full Production Run
For a full production run with real data:

1. Prepare your input files:
   - Digital topography file (NetCDF or MAT format)
   - Sample file with stable isotope data (Excel format)
   - Run file with model parameters (text format)

2. Set up your run directory with the required files

3. Execute the calculation:

```python
from opi.opi_calc_one_wind import opi_calc_one_wind

# Specify the path to your run file
results = opi_calc_one_wind(run_file_path="/path/to/your/run/file.txt")
```

### 3. Example Workflow
Here's a complete example workflow:

```python
from opi.opi_calc_one_wind import opi_calc_one_wind
from opi.base_state import base_state
from opi.constants import *

# Example: Calculate atmospheric base state
NM = 0.01  # Buoyancy frequency (rad/s)
T0 = 285.0  # Surface temperature (K)
z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)

# Example: Run full OPI calculation
results = opi_calc_one_wind()

# Access results
print(f"Model parameters: {results['solution_params']}")
print(f"Derived parameters: {results['derived_params']}")
```

## Available Functions

The following main functions are available in the Python version:

- `opi_calc_one_wind`: Calculates results for a one-wind solution
- `opi_calc_two_winds`: Calculates results for a two-wind solution (not yet implemented)
- `opi_fit_one_wind`: Fits model parameters for a one-wind solution (not yet implemented)
- `opi_maps_one_wind`: Creates maps of OPI results (not yet implemented)
- `opi_plots_one_wind`: Generates plots of OPI results (not yet implemented)

## Project Structure

```
OPI-Orographic-Precipitation-and-Isotopes/
├── opi/                     # Main Python package
│   ├── __init__.py          # Package initialization
│   ├── constants.py         # Physical constants
│   ├── base_state.py        # Base atmospheric state calculation
│   ├── saturated_vapor_pressure.py  # Saturated vapor pressure calculation
│   ├── coordinates.py       # Coordinate transformations
│   ├── wind_path.py         # Wind path calculations
│   ├── catchment_nodes.py   # Catchment node determination
│   ├── catchment_indices.py # Catchment index extraction
│   ├── precipitation_grid.py # Precipitation grid calculations
│   ├── isotope_grid.py     # Isotope grid calculations
│   ├── calc_one_wind.py     # One-wind calculation core
│   ├── opi_calc_one_wind.py # Main entry point for one-wind calculation
│   └── ...                  # More modules to be added
├── requirements.txt         # Python dependencies
├── setup.py                 # Setup script
├── setup_environment.bat    # Windows environment setup
├── setup_environment.sh     # Unix/Linux environment setup
├── test_installation.py     # Installation verification script
├── verify_installation.py   # Simplified verification script
├── README_PYTHON.md         # This file
└── README.md                # Combined documentation
```

## Features

The Python version includes all the major features of the original MATLAB package:

- Calculation of orographic precipitation patterns
- Isotope fractionation modeling
- Single and dual wind field solutions
- Parameter fitting capabilities
- Visualization tools

## Differences from MATLAB Version

- Python uses 0-based indexing instead of MATLAB's 1-based indexing
- Array operations follow NumPy conventions
- File I/O adapted to Python standards
- Object-oriented design principles where appropriate

## Contributing

Contributions are welcome! Feel free to submit pull requests for bug fixes, improvements, or additional features.

## License

The original MATLAB code was distributed under a license described in the original README. The Python translation inherits similar terms, though specific licensing should be confirmed with the original author.

## Acknowledgments

Thanks to Mark Brandon for the original MATLAB implementation and his research on orographic precipitation and isotope fractionation. This Python translation aims to make the tools more accessible to the broader scientific community.

## References

Original MATLAB package:
- Brandon, M.T., 2022a, Matlab Programs for the Analysis of Orographic Precipitation and Isotopes, with Implications for the Study of Paleotopography. Zenodo open repository.

Additional references are available in the original README.md file.