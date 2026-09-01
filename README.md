# OPI - Orographic Precipitation and Isotopes

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-blue.svg)](OPI_matlab/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](OPI_python/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive model for analyzing precipitation and water isotope fractionation associated with steady atmospheric flow over arbitrary three-dimensional topography.

## 📦 Two Versions Available

This repository contains **two implementations** of the OPI model:

| Version | Language | Status | Documentation |
|---------|----------|--------|---------------|
| **[MATLAB](OPI_matlab/)** | MATLAB R2020b+ | ✅ Original | [MATLAB Docs](OPI_matlab/README.md) |
| **[Python](OPI_python/)** | Python 3.8+ | ✅ Ported | [Python Docs](OPI_python/README.md) |

## 🎯 Quick Start

### Choose Your Version

```bash
# For MATLAB users
cd OPI_matlab/
# See OPI_matlab/README.md for instructions

# For Python users
cd OPI_python/
pip install -e ".[all]"
# See python/README.md for instructions
```

## 📚 Learning Path

### For New Users
1. Read [Getting Started](docs/getting-started.md)
2. Try the [Gaussian Mountain Example](examples/gaussian-mountain/)
3. Explore your preferred version

### For MATLAB Users Learning Python
1. Start with [MATLAB to Python Guide](docs/comparison/matlab-to-python.md)
2. Compare [side-by-side examples](examples/)
3. See [function mapping](docs/comparison/function-mapping.md)

### For Python Users
1. See [Python Quick Start](python/README.md)
2. Try [Jupyter Notebooks](python/notebooks/)
3. Explore [examples](python/examples/)

## 📂 Repository Structure

```
OPI/
├── README.md                 # This file - main entry point
├── LICENSE                   # MIT License
├── docs/                     # Shared documentation
│   ├── getting-started.md   # Getting started guide
│   ├── comparison/          # MATLAB vs Python comparison
│   └── theory/              # Theoretical background
│
├── OPI_matlab/              # MATLAB implementation
│   ├── README.md            # MATLAB-specific docs
│   ├── OPI_programs/        # Main programs
│   ├── private/             # Core functions
│   ├── data/                # Sample data
│   └── runs/                # Run files & output
│
├── OPI_python/              # Python implementation
│   ├── README.md            # Python-specific docs
│   ├── opi/                 # Main package
│   ├── tests/               # Test suite
│   ├── examples/            # Examples
│   └── notebooks/           # Jupyter notebooks
│
└── examples/                 # Shared/comparative examples
    ├── gaussian-mountain/   # Same case in both languages
    ├── himalaya/            # Real-world case study
    └── README.md            # Example guide
```

## 🔬 Side-by-Side Comparison

| Feature | MATLAB | Python |
|---------|--------|--------|
| **Installation** | MATLAB R2020b+ | `pip install -e ".[all]"` |
| **Quick Start** | Run `opiCalc_OneWind.m` | `from opi import OneWindModel` |
| **Documentation** | [MATLAB Docs](matlab/README.md) | [Python Docs](python/README.md) |
| **Examples** | [MATLAB Examples](matlab/examples/) | [Python Examples](python/examples/) |
| **Visualization** | MATLAB plots | Matplotlib + advanced viz |
| **Parallel** | Parallel Computing Toolbox | joblib, multiprocessing |
| **I/O** | `.mat` files | NumPy, NetCDF, GeoTIFF, etc. |

## 📖 Documentation

### MATLAB data assimilation

The MATLAB implementation now includes proxy data-assimilation operators,
chronology handling, posterior weighting, configurations, and tests. Start
with [`OPI_matlab/README.md`](OPI_matlab/README.md) and call
`run_assimilation(experimentRoot)` for a completed calc-only experiment.
Parallel fitting on Apple Silicon uses MATLAB's Parallel Computing Toolbox;
set the `.run` parallel flag to `1` or use `opiStartParallelPool`.

### Main Documentation
- [Getting Started](docs/getting-started.md) - Start here!
- [Installation Guide](docs/installation.md) - Detailed installation
- [Theory & Background](docs/theory/) - Physical basis

### Comparison Documentation
- [MATLAB vs Python](docs/comparison/matlab-vs-python.md) - Feature comparison
- [MATLAB to Python Guide](docs/comparison/matlab-to-python.md) - Migration guide
- [Function Mapping](docs/comparison/function-mapping.md) - Function correspondence
- [Syntax Differences](docs/comparison/syntax-differences.md) - Syntax comparison

### Version-Specific Docs
- [MATLAB Documentation](matlab/README.md)
- [Python Documentation](python/README.md)

## 🎓 Examples

### Comparative Examples (Same case in both languages)
- [Gaussian Mountain](examples/gaussian-mountain/) - Idealized case
- [Simple Ridge](examples/simple-ridge/) - 2D ridge case
- [Himalaya](examples/himalaya/) - Real-world application

### MATLAB Examples
See [matlab/examples/](matlab/examples/)

### Python Examples
See [python/examples/](python/examples/)

## 🔄 Function Mapping

| MATLAB Function | Python Function | Location |
|----------------|-----------------|----------|
| `opiCalc_OneWind` | `OneWindModel.run()` | `opi/models/` |
| `baseState` | `base_state()` | `opi/core/` |
| `fourierSolution` | `fourier_solution()` | `opi/core/` |
| `precipitationGrid` | `precipitation_grid()` | `opi/core/` |
| `isotopeGrid` | `isotope_grid()` | `opi/core/` |
| `fminCRS3` | `CRS3Optimizer` | `opi/solvers/` |

See [full mapping](docs/comparison/function-mapping.md) for complete list.

## 🛠️ Development

### Contributing
We welcome contributions to both versions! See:
- [Contributing Guide](docs/contributing.md)
- [MATLAB Contributing](matlab/CONTRIBUTING.md)
- [Python Contributing](python/CONTRIBUTING.md)

### Testing
```bash
# MATLAB
run(matlab/tests/run_tests.m)

# Python
cd python/
pytest tests/ -v
```

## 📚 Citation

If you use OPI in your research, please cite:

```bibtex
@software{opi_model,
  title = {OPI: Orographic Precipitation and Isotopes},
  author = {Brandon, Mark},
  year = {2024},
  note = {MATLAB original; Python port available}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/.../issues)
- **Discussions**: [GitHub Discussions](https://github.com/.../discussions)
- **Email**: ...

## 🙏 Acknowledgments

- **MATLAB Original**: Mark Brandon, Yale University
- **Python Port**: AI Assistant & Contributors
- **Theory**: Based on Smith & Barstad (2004), Durran & Klemp (1982)

---

**Choose your path**: [MATLAB](matlab/) | [Python](python/) | [Compare](docs/comparison/)
