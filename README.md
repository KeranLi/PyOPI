## OPI (Orographic Precipitation and Isotopes) - Python

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Python implementation of the OPI model for analyzing precipitation and isotope fractionation associated with steady atmospheric flow over topography.

### 🚀 Quick Start

```bash
# Install
pip install -e .

# Run calculation
python -m opi calc-one-wind

# Run tests
python -m opi test
```

### 📁 Project Structure

```
OPI_python/
├── opi/                    # Main package
│   ├── models/            # Data models (parameters, results, config)
│   ├── core/              # Core abstractions (calculators, optimizers)
│   ├── solvers/           # Physical solvers (FFT, precipitation, isotopes)
│   ├── calculators/       # High-level calculators
│   ├── io/                # Input/Output
│   └── plugins/           # Plugin system
├── examples/              # Example scripts
├── tests/                 # Test data and scripts
├── docs/                  # Documentation
└── README.md             # This file
```

### 🎯 Features

#### Core Physics (Phase 1 ✅)
- FFT terrain solution (Durran & Klemp 1982)
- LTOP precipitation (Smith & Barstad 2004)
- Isotope fractionation (Ciais & Jouzel 1994)
- WBF zone handling

#### Application Layer (Phase 2 ✅)
- Single wind field calculation
- Two wind fields calculation
- Parameter fitting with CRS3 optimization
- Data loading (MAT/Excel)

#### Architecture (Refactored ✅)
- Type-safe data classes
- Object-oriented design
- Plugin system
- Configuration management

### 📖 Usage

#### Command Line

```bash
# Information
python -m opi info

# Single wind calculation
python -m opi calc-one-wind [runfile]

# Two winds calculation
python -m opi calc-two-winds [runfile]

# Parameter fitting
python -m opi fit-one-wind [runfile] --iter 10000
```

#### Python API

```python
from opi import opi_calc_one_wind

# Simple usage
result = opi_calc_one_wind(verbose=True)

# With custom parameters
solution = [10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7]
result = opi_calc_one_wind(solution_vector=solution)

# Access results
precip = result['results']['precipitation']
d2h = result['results']['d2h']
```

### New Architecture (Refactored)

```python
from opi.calculators import OneWindCalculatorNew
from opi.models import OPIConfig, OneWindParameters, WindField

# Type-safe, object-oriented usage
config = OPIConfig()
calculator = OneWindCalculatorNew(config)

params = OneWindParameters(
    wind=WindField(speed=10.0, azimuth=90.0),
    ...
)

results = calculator.calculate(params)
print(results.precipitation.mean)
```

### 📚 Documentation

See the [docs/](docs/) directory for detailed documentation:

- [Functionality Analysis](docs/FUNCTIONALITY_GAP_ANALYSIS.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Completion Report](docs/FINAL_COMPLETION_REPORT.md)
- [Refactoring Guide](docs/REFACTORING_SUMMARY.md)

### 🔧 Installation

#### Requirements
- Python 3.8+, NumPy, SciPy, Matplotlib,  Pandas, XArray (optional)

### Install

```bash
cd OPI_python
pip install -e .
```

Or with optional dependencies:

```bash
pip install -e ".[netcdf]"
```

### 🧪 Testing

```bash
# Run all tests
python -m opi test

# Or individually
python tests/test_installation.py
python tests/verify_installation.py
```

### 📊 Examples

See [examples/](examples/) directory:

- `comprehensive_example.py` - Full feature demo
- `complete_workflow_example.py` - Workflow demonstration
- `single_wind_example.py` - Basic usage

Run example:

```bash
cd examples
python comprehensive_example.py
```

### 🏗️ Architecture

The project has two architectures:

#### Legacy (MATLAB-style)
Direct translation of MATLAB code. Simple but less structured.

#### Refactored (Pythonic)
New object-oriented architecture with:
- Data classes for type safety
- Abstract base classes for extensibility
- Plugin system for custom components
- Configuration management

See [docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md) for details.

### 📝 Citation

Original MATLAB implementation:
- Brandon, M.T., 2022. Matlab Programs for the Analysis of Orographic Precipitation and Isotopes.

Python implementation:
-Keran Li, 2026. OPI Python Port.

### 📧 Contact

Original Author: Mark Brandon (Yale University)  
Python Port: Keran Li (Nanjing University)

## 📄 License

See [LICENSE](../LICENSE) file for details.
