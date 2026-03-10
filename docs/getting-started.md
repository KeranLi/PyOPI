# Getting Started with OPI

Welcome! This guide will help you get started with the OPI (Orographic Precipitation and Isotopes) model.

## 🚀 Choose Your Path

### I want to use MATLAB
→ Go to [MATLAB Version](../OPI_matlab/README.md)

### I want to use Python
→ Go to [Python Version](../OPI_python/README.md)

### I want to see both versions
→ Continue reading below

---

## 📋 Prerequisites

### For MATLAB
- MATLAB R2020b or later
- No additional toolboxes required

### For Python
- Python 3.8 or later
- See `requirements.txt`

## 📦 Installation

### MATLAB
```matlab
% No installation needed - just add to path
addpath('OPI_matlab/OPI_programs');
```

### Python
```bash
cd OPI_python
pip install -e ".[all]"
```

## 🎯 Quick Example

### Gaussian Mountain Case

We provide the same example in both languages:

#### MATLAB
```matlab
cd OPI_matlab/OPI_programs
% Load example run file and execute
opiCalc_OneWind
```

#### Python
```python
cd OPI_python
python examples/simple_example.py
```

Results should be nearly identical!

## 📚 Next Steps

### Learn the Theory
- Read [Theory Background](theory/background.md)
- Understand the [physical basis](theory/physical-basis.md)

### Compare Implementations
- See [Function Mapping](comparison/function-mapping.md)
- Read [MATLAB to Python Guide](comparison/matlab-to-python.md)

### Run Examples
- Try [Gaussian Mountain](../examples/gaussian-mountain/)
- Try [Simple Ridge](../examples/simple-ridge/)

## ❓ Common Questions

### Which version should I use?

**Use MATLAB if:**
- You're already familiar with MATLAB
- Your lab/group uses MATLAB
- You need the original reference implementation

**Use Python if:**
- You want open-source tools
- You need advanced visualization
- You want parallel processing
- You need multiple I/O formats

**Use both if:**
- You're validating results
- You're learning both languages
- You're comparing implementations

### Are the results identical?

Nearly! Small numerical differences (< 0.1%) are normal due to:
- Different FFT implementations
- Floating-point precision
- Random number generation

The physics is identical.

### Can I convert my code?

Yes! See the [migration guide](comparison/matlab-to-python.md).

## 📞 Getting Help

- **MATLAB issues**: Check [MATLAB README](../OPI_matlab/README.md)
- **Python issues**: Check [Python README](../OPI_python/README.md)
- **General questions**: See [Project README](../README.md)

## 🎓 Tutorial Series

1. [Quick Start](tutorials/01_quick_start.md)
2. [Single Wind Simulation](tutorials/02_single_wind_simulation.md)
3. [Two Wind Simulation](tutorials/03_two_winds_simulation.md)
4. [Parameter Fitting](tutorials/04_parameter_fitting.md)
5. [Working with Real Data](tutorials/05_working_with_real_data.md)

---

**Ready?** Choose your version: [MATLAB](../OPI_matlab/README.md) | [Python](../OPI_python/README.md)
