# OPI Python - New Features for MATLAB Compatibility

This document describes the new features added to the OPI Python package to fully replicate the MATLAB OPI 3.7 functionality.

## Overview

The Python OPI package now supports:

1. **Complete .run file workflow** - Parse MATLAB-style .run files and execute simulations
2. **One-wind and Two-wind models** - Full support for both moisture source configurations
3. **Parameter fitting** - CRS3 global optimization for parameter estimation
4. **Visualization** - Generate publication-quality maps and figures
5. **MATLAB-compatible output** - Save results in .mat format for interoperability

## Main Entry Points

### 1. Run Simulation from .run File

```python
from opi import run_simulation

# Run simulation using .run file configuration
result = run_simulation('runs/run001/run001.run', verbose=True)

if result['success']:
    print(f"Results saved to: {result['results_path']}")
    print(f"Log saved to: {result['log_path']}")
```

Or from command line:
```bash
python -m opi run runs/run001/run001.run
```

### 2. Parameter Fitting

```python
from opi import opi_fit_one_wind, opi_fit_two_winds

# Fit one-wind model
result = opi_fit_one_wind('runs/run001/run001.run', max_iterations=5000)

# Fit two-wind model
result = opi_fit_two_winds('runs/run001/run001.run', max_iterations=10000)
```

Or from command line:
```bash
python -m opi fit-one runs/run001/run001.run --max-iter 5000
python -m opi fit-two runs/run001/run001.run --max-iter 10000
```

### 3. Generate Maps

```python
from opi import opi_maps_one_wind, opi_maps_two_winds

# Generate maps from results
files = opi_maps_one_wind('runs/run001/opiCalc_OneWind_Results.mat')
files = opi_maps_two_winds('runs/run001/opiCalc_TwoWinds_Results.mat')
```

Or from command line:
```bash
python -m opi maps-one runs/run001/opiCalc_OneWind_Results.mat
python -m opi maps-two runs/run001/opiCalc_TwoWinds_Results.mat
```

## Run File Format

The .run file format matches the MATLAB OPI specification:

```
Line 1:  Run title
Line 2:  Parallel flag (0 or 1)
Line 3:  Primary data path
Line 4:  Alternate data path (or 'no')
Line 5:  Topography file (e.g., topography.mat)
Line 6:  Tukey window size (0-1)
Line 7:  Sample file (e.g., samples.xlsx, or 'no')
Line 8:  Continental divide file (e.g., divide.mat, or 'no')
Line 9:  Map limits [minLon, maxLon, minLat, maxLat]
Line 10: Section origin (lon lat, or 'map')
Line 11: CRS3 parameters (mu epsilon)
Line 12: Restart file ('no' if none)
Line 13: Parameter labels (9 or 19 labels separated by |)
Line 14: Exponents (9 or 19 values)
Line 15: Lower bounds (9 or 19 values)
Line 16: Upper bounds (9 or 19 values)
Line 17: Solution vector (optional, for opiCalc only)
```

### One-Wind Example

```
Himalaya Test Run 001
0
F:\OPI\data
no
topography.mat
0.5
samples.xlsx
no
77.5, 97.5, 20.0, 36.0
90.0, 29.0
20, 1e-4
no
U|Azimuth|T0|M|kappa|tauC|d2H0|dLat|fP
0, 0, 0, 0, 0, 0, 0, 0, 0
0.1, -90, 270, 0, 0, 0, -0.060, -0.015, 0
20, 90, 300, 1, 1e6, 2500, 0, 0, 1
6.6, 33.8, 293.5, 0.34, 233368, 1296, -0.0026, -0.000533, 1.0
```

### Two-Wind Example

```
Himalaya Two-Wind Test
0
F:\OPI\data
no
topography.mat
0.5
samples.xlsx
divide.mat
77.5, 97.5, 20.0, 36.0
90.0, 29.0
20, 1e-4
no
U|Azimuth|T0|M|kappa|tauC|d2H0|dLat|fP|fraction|U2|Az2|T02|M2|kappa2|tauC2|d2H0_2|dLat2|fP2
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
0.1, -90, 270, 0, 0, 0, -0.060, -0.015, 0, 0, 0.1, 0, 270, 0, 0, 0, -0.060, -0.015, 0
20, 90, 300, 1, 1e6, 2500, 0, 0, 1, 1, 20, 180, 300, 1, 1e6, 2500, 0, 0, 1
6.6, 33.8, 293.5, 0.34, 233368, 1296, -0.0026, -0.000533, 1.0, 0.52, 16.0, 125.7, 294.0, 0.693, 722120, 2450, -0.0001, -0.000007, 1.0
```

## Two-Wind Model Specifics

The two-wind model requires:

1. **Continental divide file** (.mat format with `contDivideLon` and `contDivideLat`)
2. **19 parameters** (9 for wind 1, 9 for wind 2, 1 for fraction)
3. **Sample classification** - Samples are classified by which side of the divide they fall on

The divide file defines a polyline that separates the study area into two sides, each associated with one wind field.

## Output Files

### Simulation Results (.mat format)

The results file contains:
- `beta` - Solution vector (9 or 19 parameters)
- `chiR2` - Reduced chi-square value
- `nu` - Degrees of freedom
- `pGrid` - Precipitation rate grid
- `d2HGrid`, `d18OGrid` - Isotope grids
- `lon`, `lat`, `x`, `y` - Coordinate grids
- Sample predictions (if samples provided)

For two-wind results:
- `pGrid_1`, `pGrid_2` - Individual precipitation grids
- `d2HGrid_1`, `d2HGrid_2` - Individual isotope grids
- `fractionPGrid` - Fraction from wind field 1

### Log Files

Text files containing:
- Run information and parameters
- Computation progress
- Final statistics
- Timing information

### Visualization Output

PDF files including:
- Topography and sample locations
- Precipitation distribution
- Isotope distribution maps
- Parameter summary tables
- Comparison plots (two-wind)

## API Reference

### Core Functions

#### `run_simulation(run_file_path, verbose=True)`
Main entry point for running simulations from .run files.

#### `opi_calc_one_wind(run_file_path=None, solution_vector=None, verbose=True)`
Calculate one-wind model results.

#### `opi_calc_two_winds(run_file_path=None, solution_vector=None, verbose=True)`
Calculate two-wind model results.

#### `opi_fit_one_wind(run_file_path, verbose=True, max_iterations=10000)`
Fit one-wind model parameters using CRS3 optimization.

#### `opi_fit_two_winds(run_file_path, verbose=True, max_iterations=10000)`
Fit two-wind model parameters using CRS3 optimization.

#### `opi_maps_one_wind(results_file, save_plots=True, show_plots=False)`
Generate visualization maps for one-wind results.

#### `opi_maps_two_winds(results_file, save_plots=True, show_plots=False)`
Generate visualization maps for two-wind results.

### Utility Functions

#### `parse_run_file(run_file_path)`
Parse a .run configuration file.

#### `get_input(data_path, topo_file, r_tukey, sample_file, sd_res_ratio)`
Load topography and sample data.

## Examples

See the `examples/` directory for complete working examples:

- `run_simulation_example.py` - Run simulations from .run files
- `fitting_example.py` - Parameter fitting examples
- `complete_workflow_example.py` - End-to-end workflow

## Command-Line Interface

The package provides a unified CLI:

```bash
# Show help
python -m opi --help

# Show command-specific help
python -m opi run --help
python -m opi fit-one --help

# Run commands
python -m opi run config.run
python -m opi fit-one config.run --max-iter 5000
python -m opi maps-one results.mat
```

## MATLAB Compatibility Notes

### What's Compatible

- .run file format (identical to MATLAB)
- Output .mat file format (readable by MATLAB)
- Physical calculations (same algorithms)
- CRS3 optimization algorithm

### What's Different

- Plotting style (matplotlib vs MATLAB graphics)
- Log file format (similar but not identical)
- Some internal variable naming

### Migrating from MATLAB

To migrate from MATLAB OPI:

1. Use the same .run files (no changes needed)
2. Results are saved in the same .mat format
3. Plotting produces similar but not identical figures
4. Performance may differ (Python vs MATLAB optimization)

## Troubleshooting

### Common Issues

1. **"Continental divide file required for two-wind"**
   - Two-wind model requires a divide file. Create a .mat file with `contDivideLon` and `contDivideLat` variables.

2. **"Sample file is required for fitting"**
   - Parameter fitting requires observational data. Ensure the run file specifies a valid sample file.

3. **"Optimization failed"**
   - Check parameter bounds are reasonable
   - Increase max_iterations
   - Check sample data quality

4. **Import errors**
   - Install required packages: `pip install numpy scipy matplotlib`
   - For two-wind divide processing: `pip install shapely`

## Version Information

- OPI Python Version: 1.0.0
- Compatible with MATLAB OPI: 3.7
- Last updated: 2026-03-09
