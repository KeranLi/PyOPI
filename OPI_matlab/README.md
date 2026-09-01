# OPI - MATLAB Simulation and Data Assimilation

This directory contains the OPI forward model, fitting and mapping programs,
plus proxy data-assimilation operators and tests.

For assimilation, call `run_assimilation(experimentRoot)` after generating
calc-only OPI results. The experiment must contain
`design/case_manifest.csv` and `calc_only/<case_id>/` result files.

## Parallel computation

The fitting code supports local CPU parallelism through MATLAB's Parallel
Computing Toolbox. Set the second non-comment line of a `.run` file to `1` to
let `fminCRS3` evaluate candidate solutions in parallel; leave it at `0` for
serial execution. The bundled `runs/run033.run` template is configured for
parallel execution. On Apple Silicon, start with four to eight process workers:

```matlab
addpath('OPI_programs')
pool = opiStartParallelPool(6);
```

`opiStartParallelPool` reuses or resizes the local pool. The default worker
count is conservative (`min(number_of_cores - 1, 8)`). Close it when finished
with `delete(pool)`. Parallel execution requires Parallel Computing Toolbox;
the core OPI model remains usable without it in serial mode.

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-blue.svg)]()

Original MATLAB implementation of the OPI (Orographic Precipitation and Isotopes) model.

## 📋 Contents

- Main OPI programs
- Core computational functions (private/)
- Sample data
- Example runs

## 🚀 Quick Start

### Prerequisites
- MATLAB R2020b or later
- Parallel Computing Toolbox is optional for serial simulation and required
  for parallel fitting.

### Running a Simulation

1. **Navigate to OPI_programs directory**
   ```matlab
   cd OPI_programs
   ```

2. **Run a calculation**
   ```matlab
   % Single wind simulation
   opiCalc_OneWind
   
   % Or with two winds
   opiCalc_TwoWinds
   ```

3. **View results**
   ```matlab
   % Load results
   load('opiCalc_OneWind_Results.mat');
   
   % Plot precipitation
   contourf(pGrid);
   colorbar;
   title('Precipitation Rate');
   ```

## 📂 Directory Structure

```
OPI_matlab/
├── README.md                   # This file
├── OPI_programs/              # Main programs
│   ├── private/               # Core functions
│   │   ├── baseState.m       # Base state calculation
│   │   ├── fourierSolution.m # Fourier solution
│   │   ├── precipitationGrid.m
│   │   ├── isotopeGrid.m
│   │   └── ...
│   ├── opiCalc_OneWind.m      # One-wind simulation
│   ├── opiCalc_TwoWinds.m     # Two-wind simulation
│   ├── opiFit_OneWind.m       # Parameter fitting
│   └── opiFit_TwoWinds.m
├── data/                      # Sample data
│   ├── *.xlsx                # Sample data files
│   └── ...
└── runs/                      # Example runs
    └── *.run                 # Run configuration files
```

## 📖 Main Programs

### Simulation Programs

| Program | Purpose |
|---------|---------|
| `opiCalc_OneWind` | Single moisture source simulation |
| `opiCalc_TwoWinds` | Two moisture sources simulation |

### Fitting Programs

| Program | Purpose |
|---------|---------|
| `opiFit_OneWind` | Fit parameters to data (one wind) |
| `opiFit_TwoWinds` | Fit parameters to data (two winds) |

### Plotting Programs

| Program | Purpose |
|---------|---------|
| `opiPlots_OneWind` | Generate plots for one-wind results |
| `opiMaps_OneWind` | Generate maps for one-wind results |

## 🔧 Core Functions (private/)

### Atmospheric Calculations
- `baseState.m` - Atmospheric base state
- `saturatedVaporPressure.m` - Vapor pressure

### Flow Solutions
- `fourierSolution.m` - Fourier solution for airflow
- `windGrid.m` - Coordinate transformation

### Precipitation
- `precipitationGrid.m` - Precipitation calculation
- `isotherm.m` - Isotherm height

### Isotopes
- `isotopeGrid.m` - Isotope calculations
- `fractionationHydrogen.m` - H fractionation
- `fractionationOxygen.m` - O fractionation

### Utilities
- `lonlat2xy.m` / `xy2lonlat.m` - Coordinate conversion
- `catchmentNodes.m` / `catchmentIndices.m` - Catchment utilities
- `lifting.m` / `windPath.m` - Lifting calculations
- `fminCRS3.m` - Optimization algorithm

## 📊 Example Usage

### Basic Simulation
```matlab
% Navigate to programs directory
cd OPI_programs

% Run single wind simulation
opiCalc_OneWind

% Follow prompts to select run file
% Results saved automatically
```

### Parameter Fitting
```matlab
% Run fitting with one wind
opiFit_OneWind

% Follow prompts to select:
% - Data file
% - Run configuration
% - Parameter bounds
```

### Custom Calculation
```matlab
% Load your data
load('my_topography.mat');

% Define parameters
beta = [10, 270, 288, 0.8, 0, 1000, -0.1, 0, 1];
fC = 1e-4;
hR = 540;

% Run calculation
[chiR2, nu, stdResiduals, ...] = calc_OneWind(beta, fC, hR, ...);

% Plot results
figure;
subplot(1,2,1);
contourf(pGrid);
title('Precipitation');
colorbar;

subplot(1,2,2);
contourf(d2HGrid * 1000);
title('d2H (permil)');
colorbar;
```

## 🎓 Learning Resources

### Documentation
- Function help: `help functionName`
- Program help: `help opiCalc_OneWind`

### Examples
See `runs/` directory for example run files and outputs.

### Comparison with Python
- [Function Mapping](../docs/comparison/function-mapping.md)
- [MATLAB to Python Guide](../docs/comparison/matlab-to-python.md)

## 🔄 Python Version Available

A complete Python port is available:
- [Python Version](../python/README.md)
- Same algorithms, modern Python features
- More I/O formats, better visualization

## 📝 Citation

```bibtex
@software{opi_matlab,
  title = {OPI: Orographic Precipitation and Isotopes},
  author = {Brandon, Mark},
  institution = {Yale University},
  year = {2024}
}
```

## 📞 Support

For issues specific to the MATLAB version:
- Check function documentation: `help functionName`
- Review example run files in `runs/`
- Compare with working examples

## 🙏 Acknowledgments

- Mark Brandon, Yale University - Original implementation
- Theory based on Smith & Barstad (2004), Durran & Klemp (1982)

---

**See Also**: [Python Version](../python/) | [Comparison](../docs/comparison/)
