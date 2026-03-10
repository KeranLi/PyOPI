# OPI Examples

This directory contains example cases for the OPI model, with implementations in both **MATLAB** and **Python**.

## 📂 Directory Structure

```
examples/
├── README.md                          # This file
│
├── gaussian-mountain/                 # Idealized Gaussian mountain
│   ├── README.md                      # Case description
│   ├── matlab/                        # MATLAB implementation
│   │   ├── run_simulation.m
│   │   └── plot_results.m
│   └── python/                        # Python implementation
│       ├── run_simulation.py
│       └── plot_results.py
│
├── simple-ridge/                      # 2D ridge case
│   ├── README.md
│   ├── matlab/
│   └── python/
│
└── himalaya/                          # Real-world Himalaya case
    ├── README.md
    ├── matlab/
    └── python/
```

## 🎯 Purpose

These examples serve to:

1. **Demonstrate OPI usage** in both languages
2. **Compare implementations** side-by-side
3. **Validate results** between MATLAB and Python
4. **Learn by example** - see the same case in both languages

## 📖 Available Examples

### 1. Gaussian Mountain

**Idealized case with analytical solution**

- Simple Gaussian topography
- Single wind regime
- Perfect for learning basics

[Go to example →](gaussian-mountain/)

### 2. Simple Ridge

**2D ridge with uniform flow**

- Infinite ridge (2D approximation)
- Compare with linear theory
- Good for understanding 2D vs 3D

[Go to example →](simple-ridge/)

### 3. Himalaya Case

**Real-world application**

- Actual topography data
- Real observational constraints
- Complex multi-wind scenario

[Go to example →](himalaya/)

## 🔬 Comparison Methodology

Each example includes:

1. **Same input data** - Identical topography and parameters
2. **Equivalent workflows** - Same calculation steps
3. **Result comparison** - Validate outputs match
4. **Performance notes** - Timing and resource usage

## 🚀 Running Examples

### MATLAB
```matlab
% Navigate to example directory
cd examples/gaussian-mountain/matlab

% Run the example
run_simulation

% Plot results
plot_results
```

### Python
```bash
# Navigate to example directory
cd examples/gaussian-mountain/python

# Run the example
python run_simulation.py

# Plot results
python plot_results.py
```

## 📊 Expected Results

All examples should produce **equivalent results** in both languages (within numerical precision).

Typical outputs:
- Precipitation rate maps
- Isotope distribution (δ²H, δ¹⁸O)
- Cross-sections
- Statistical comparisons

## 🎓 Learning Path

### For Beginners
1. Start with **Gaussian Mountain** (simplest)
2. Compare MATLAB and Python versions
3. Understand the workflow

### For Experienced Users
1. Try **Himalaya** case
2. Modify parameters
3. Compare with observations

### For Researchers
1. Use as template for your own case
2. Adapt topography data
3. Apply to your study region

## 📝 Notes

### Numerical Differences
Small differences between MATLAB and Python results are normal due to:
- Different FFT implementations
- Numerical precision differences
- Random number generation (for fitting)

Typical differences:
- Precipitation: < 0.1%
- Isotopes: < 0.01‰
- Timing: Depends on system

### File Formats

**MATLAB**: `.mat` files
```matlab
load('results.mat');
```

**Python**: `.npz` or `.nc` files
```python
import numpy as np
data = np.load('results.npz')
# or
import xarray as xr
data = xr.open_dataset('results.nc')
```

## 🤝 Contributing

Have an interesting case study? Consider contributing:
1. Create a new subdirectory
2. Include both MATLAB and Python versions
3. Add README with description
4. Submit a pull request

## 📚 See Also

- [MATLAB Documentation](../matlab/README.md)
- [Python Documentation](../python/README.md)
- [Function Mapping](../docs/comparison/function-mapping.md)
- [MATLAB vs Python](../docs/comparison/matlab-vs-python.md)
