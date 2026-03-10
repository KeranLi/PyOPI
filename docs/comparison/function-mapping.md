# MATLAB to Python Function Mapping

This document provides a complete mapping between MATLAB and Python implementations of the OPI model.

## Core Functions

### Atmospheric Base State

| MATLAB | Python | Description |
|--------|--------|-------------|
| `baseState(NM, T0)` | `base_state(NM, T0)` | Calculate atmospheric base state |
| `saturatedVaporPressure(T)` | `saturated_vapor_pressure(T)` | Vapor pressure calculation |

**MATLAB**:
```matlab
[zBar, T, gammaEnv, gammaSat, gammaRatio, rhoS0, hS, rho0, hRho] = baseState(NM, T0);
```

**Python**:
```python
from opi.core import base_state

result = base_state(NM=0.01, T0=288)
z_bar = result['z_bar']
T = result['T']
gamma_env = result['gamma_env']
# ... etc
```

### Fourier Solution

| MATLAB | Python | Description |
|--------|--------|-------------|
| `fourierSolution(...)` | `fourier_solution(...)` | Fourier solution for airflow |
| `windGrid(x, y, azimuth)` | `wind_grid(x, y, azimuth)` | Coordinate transformation |

**MATLAB**:
```matlab
[s, t, Sxy, Txy, hWind, kS, kT, hHat, kZ] = fourierSolution(x, y, hGrid, U, azimuth, NM, fC, hRho);
```

**Python**:
```python
from opi.core import fourier_solution

result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
s = result['s']
t = result['t']
Sxy = result['Sxy']
# ... etc
```

### Precipitation

| MATLAB | Python | Description |
|--------|--------|-------------|
| `precipitationGrid(...)` | `precipitation_grid(...)` | Precipitation rate calculation |
| `isotherm(TR, ...)` | `isotherm(TR, ...)` | Isotherm height calculation |

**MATLAB**:
```matlab
[s, t, Sxy, Txy, pGrid, hWind, fMWind, rHWind, fPWind, z223Wind, z258Wind, tauF] = ...
    precipitationGrid(x, y, hGrid, U, azimuth, NM, fC, kappa, tauC, hRho, zBar, T, ...
    gammaEnv, gammaSat, gammaRatio, rhoS0, hS, fP0);
```

**Python**:
```python
from opi.core import precipitation_grid

result = precipitation_grid(x, y, h_grid, U, azimuth, NM, f_c, kappa, tau_c, 
                           h_rho, z_bar, T, gamma_env, gamma_sat, gamma_ratio,
                           rho_s0, h_s, f_p0)
p_grid = result['p_grid']
tau_f = result['tau_f']
# ... etc
```

### Isotopes

| MATLAB | Python | Description |
|--------|--------|-------------|
| `isotopeGrid(...)` | `isotope_grid(...)` | Isotope grid calculation |
| `fractionationHydrogen(T)` | `fractionation_hydrogen(T)` | H fractionation |
| `fractionationOxygen(T)` | `fractionation_oxygen(T)` | O fractionation |

## High-Level Models

### Single Wind Model

| MATLAB | Python | Description |
|--------|--------|-------------|
| `opiCalc_OneWind` | `OneWindModel` | One-wind simulation |
| `opiFit_OneWind` | `OneWindModel.run()` with fitting | Parameter fitting |
| `calc_OneWind` | Internal calculation | Core calculation |

**MATLAB**:
```matlab
% Run simulation
opiCalc_OneWind

% Inside the script:
[chiR2, nu, stdResiduals, ...] = calc_OneWind(beta, fC, hR, x, y, ...);
```

**Python**:
```python
from opi import OneWindModel, OneWindParameters

# Create model
model = OneWindModel(topography)

# Define parameters
params = OneWindParameters(
    U=10.0, azimuth=270.0, T0=288.0, M=0.8,
    kappa=0.0, tau_c=1000.0, d2H0=-0.1,
    d_d2H0_d_lat=0.0, f_p0=1.0
)

# Run simulation
result = model.run(params)

# Access results
print(result.p_grid)
print(result.d2H_grid)
```

### Two Winds Model

| MATLAB | Python | Description |
|--------|--------|-------------|
| `opiCalc_TwoWinds` | `TwoWindModel` | Two-wind simulation |
| `opiFit_TwoWinds` | `TwoWindModel.run()` with fitting | Parameter fitting |
| `calc_TwoWinds` | Internal calculation | Core calculation |

## Utilities

### Coordinate Transformations

| MATLAB | Python | Description |
|--------|--------|-------------|
| `lonlat2xy(lon, lat, lon0, lat0)` | `lonlat2xy(lon, lat, lon0, lat0)` | Lon/lat to x/y |
| `xy2lonlat(x, y, lon0, lat0)` | `xy2lonlat(x, y, lon0, lat0)` | x/y to lon/lat |

### Catchment Utilities

| MATLAB | Python | Description |
|--------|--------|-------------|
| `catchmentNodes(...)` | `catchment_nodes(...)` | Find catchment nodes |
| `catchmentIndices(...)` | `catchment_indices(...)` | Extract indices |

### Lifting

| MATLAB | Python | Description |
|--------|--------|-------------|
| `lifting(...)` | `lifting(...)` | Calculate lifting |
| `windPath(...)` | `wind_path(...)` | Generate wind path |

## Optimization

| MATLAB | Python | Description |
|--------|--------|-------------|
| `fminCRS3(...)` | `CRS3Optimizer` | Global optimization |
| `fminCRS3(fun, x0, ...)` | `fmin_crs3(fun, x0, ...)` | Convenience function |

**MATLAB**:
```matlab
[xBest, fBest, nIter, nEval] = fminCRS3(fun, x0, lB, uB, options);
```

**Python**:
```python
from opi.solvers import CRS3Optimizer, fmin_crs3

# Using optimizer object
optimizer = CRS3Optimizer(bounds=(lower_bounds, upper_bounds))
result = optimizer.minimize(objective_function, x0)

# Using convenience function
result = fmin_crs3(objective_function, x0, bounds=(lower_bounds, upper_bounds))
```

## Key Differences

### 1. Output Handling

**MATLAB**: Multiple output arguments
```matlab
[a, b, c] = function(x, y);
```

**Python**: Dictionary or object return
```python
result = function(x, y)
a = result['a']  # or result.a if using dataclass
```

### 2. Indexing

**MATLAB**: 1-based indexing
```matlab
value = array(1);  % First element
```

**Python**: 0-based indexing
```python
value = array[0]  # First element
```

### 3. Arrays

**MATLAB**: Column-major, meshgrid format
```matlab
[X, Y] = meshgrid(x, y);  % X varies across columns
```

**Python**: Row-major, similar meshgrid
```python
X, Y = np.meshgrid(x, y)  # X varies across columns (same!)
```

### 4. Function Names

**MATLAB**: camelCase
```matlab
functionName(input)
```

**Python**: snake_case
```python
function_name(input)
```

## Complete Workflow Comparison

### MATLAB Workflow
```matlab
% 1. Load topography
load('topo.mat');

% 2. Define parameters
beta = [10, 270, 288, 0.8, 0, 1000, -0.1, 0, 1];

% 3. Run simulation
opiCalc_OneWind;

% 4. Plot results
figure; contourf(pGrid);
```

### Python Workflow
```python
# 1. Load topography
from opi.io import load_topography
topo = load_topography('topo.nc')

# 2. Define parameters
from opi import OneWindParameters
params = OneWindParameters(
    U=10, azimuth=270, T0=288, M=0.8,
    kappa=0, tau_c=1000, d2H0=-0.1,
    d_d2H0_d_lat=0, f_p0=1.0
)

# 3. Run simulation
from opi import OneWindModel
model = OneWindModel(topo)
result = model.run(params)

# 4. Plot results
from opi.visualization import plot_precipitation_contour
plot_precipitation_contour(topo['x'], topo['y'], result.p_grid)
```

## Migration Tips

1. **Start with high-level models**: Use `OneWindModel` instead of individual functions
2. **Check dictionary keys**: Python returns dictionaries, MATLAB returns multiple outputs
3. **Adjust indexing**: Remember Python uses 0-based indexing
4. **Use visualization module**: Python has enhanced plotting capabilities
5. **Leverage I/O module**: Python supports more file formats

## See Also

- [MATLAB to Python Guide](matlab-to-python.md)
- [Syntax Differences](syntax-differences.md)
- [Python Package Documentation](../../python/README.md)
- [MATLAB Package Documentation](../../matlab/README.md)
