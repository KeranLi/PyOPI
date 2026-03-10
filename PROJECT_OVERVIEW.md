# OPI Project Overview

## Project Mission

Provide **dual-language implementations** of the OPI (Orographic Precipitation and Isotopes) model to maximize accessibility for researchers in atmospheric science, hydrology, and related fields.

## 🌐 Two Implementations, One Model

```
┌─────────────────────────────────────────────────────────────┐
│                    OPI Model Core                           │
│         (Physical Theory & Algorithms)                      │
│  Based on Smith & Barstad (2004), Durran & Klemp (1982)     │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│    MATLAB Version     │       │    Python Version     │
├───────────────────────┤       ├───────────────────────┤
│ • Original reference  │       │ • Modern ecosystem    │
│ • Academic standard   │       │ • Open source         │
│ • Proven codebase     │       │ • Enhanced features   │
└───────────────────────┘       └───────────────────────┘
```

## 📂 Repository Structure

```
OPI/
│
├── 📄 README.md                    ← Start here! Main entry point
├── 📄 PROJECT_OVERVIEW.md          ← This file - architecture guide
├── 📄 LICENSE                       ← MIT License
│
├── 📁 docs/                         ← Unified documentation
│   ├── getting-started.md          ← New user guide
│   ├── installation.md             ← Installation instructions
│   ├── comparison/                 ← MATLAB vs Python comparison
│   │   ├── function-mapping.md     ← Complete function correspondence
│   │   ├── matlab-to-python.md     ← Migration guide
│   │   └── syntax-differences.md   ← Syntax comparison
│   └── theory/                     ← Theoretical documentation
│
├── 📁 matlab/                       ← MATLAB Implementation
│   ├── README.md                   ← MATLAB-specific documentation
│   ├── OPI_programs/               ← Main programs
│   │   ├── private/                ← Core functions (original)
│   │   ├── opiCalc_OneWind.m
│   │   ├── opiCalc_TwoWinds.m
│   │   └── ...
│   ├── data/                       ← Sample data
│   └── runs/                       ← Example runs
│
├── 📁 python/                       ← Python Implementation
│   ├── README.md                   ← Python-specific documentation
│   ├── opi/                        ← Main package
│   │   ├── core/                   ← Core computations
│   │   ├── models/                 ← High-level models
│   │   ├── solvers/                ← Optimization
│   │   ├── io/                     ← I/O utilities
│   │   ├── utils/                  ← Utilities
│   │   ├── visualization/          ← Advanced plotting
│   │   └── parallel/               ← Parallel processing
│   ├── tests/                      ← Test suite (87+ tests)
│   ├── examples/                   ← Python examples
│   └── notebooks/                  ← Jupyter notebooks
│
├── 📁 examples/                     ← Shared/comparative examples
│   ├── gaussian-mountain/          ← Same case in both languages
│   │   ├── matlab/                 ← MATLAB version
│   │   └── python/                 ← Python version
│   ├── simple-ridge/
│   └── himalaya/
│
└── 📁 notebooks/                    ← Jupyter notebooks (Python)
    ├── 01_quick_start.ipynb
    ├── 02_parameter_fitting.ipynb
    └── ...
```

## 🎯 Use Cases

### I have MATLAB and want to...
- **Use OPI immediately**: Go to `matlab/OPI_programs/`
- **Understand the physics**: Read `docs/theory/`
- **Try the Python version**: See `python/README.md`

### I have Python and want to...
- **Use OPI immediately**: Go to `python/`, run `pip install -e ".[all]"`
- **Learn OPI**: Start with `notebooks/01_quick_start.ipynb`
- **Understand MATLAB origins**: See `matlab/README.md`

### I want to compare both versions...
- **See examples**: Go to `examples/gaussian-mountain/`
- **Function mapping**: Read `docs/comparison/function-mapping.md`
- **Migrate code**: See `docs/comparison/matlab-to-python.md`

## 🔑 Key Principles

### 1. Algorithmic Equivalence
Both implementations use **identical algorithms**:
- Same physical equations
- Same numerical methods
- Same default parameters

### 2. Result Consistency
Results should be **numerically equivalent**:
- Precipitation rates: < 0.1% difference
- Isotope values: < 0.01‰ difference
- Timing: Depends on system

### 3. Clear Documentation
- Each version has its own README
- Unified comparison documentation
- Side-by-side examples

### 4. Independent but Connected
- Each version can be used standalone
- Documentation links between versions
- Examples show both implementations

## 📊 Feature Comparison

| Feature | MATLAB | Python | Notes |
|---------|--------|--------|-------|
| **Core Model** | ✅ | ✅ | Identical algorithms |
| **Installation** | MATLAB R2020b+ | `pip install` | Python easier for new users |
| **I/O Formats** | `.mat` | `.npz`, `.nc`, `.tif`, `.mat` | Python more formats |
| **Visualization** | MATLAB plots | Matplotlib + advanced | Python more options |
| **Parallel** | Parallel Toolbox | joblib, multiprocessing | Python more flexible |
| **Testing** | Manual | pytest (87+ tests) | Python better coverage |
| **Documentation** | Function help | Sphinx + notebooks | Python more extensive |
| **Extensibility** | Good | Excellent | Python easier to extend |

## 🚀 Quick Start by Language

### MATLAB Users
```matlab
% Navigate to MATLAB directory
cd matlab/OPI_programs

% Run example
opiCalc_OneWind
```

### Python Users
```bash
# Navigate to Python directory
cd python

# Install
pip install -e ".[all]"

# Run example
python examples/simple_example.py
```

## 📖 Documentation Structure

### For All Users
- `README.md` - Main entry point
- `docs/getting-started.md` - Getting started guide
- `docs/installation.md` - Installation instructions

### For MATLAB Users
- `matlab/README.md` - MATLAB documentation
- `help functionName` - Function documentation

### For Python Users
- `python/README.md` - Python documentation
- `docs/api/` - API reference
- `notebooks/` - Interactive tutorials

### For Comparing
- `docs/comparison/function-mapping.md` - Function correspondence
- `docs/comparison/matlab-to-python.md` - Migration guide
- `examples/` - Side-by-side examples

## 🎓 Recommended Learning Paths

### Path 1: MATLAB → Python
1. Familiar with MATLAB version
2. Read `docs/comparison/matlab-to-python.md`
3. Try `examples/gaussian-mountain/`
4. See `python/notebooks/01_quick_start.ipynb`

### Path 2: Python Only
1. Start with `python/README.md`
2. Run `python/examples/simple_example.py`
3. Explore `python/notebooks/`
4. (Optional) Compare with MATLAB

### Path 3: Research Application
1. Read `docs/theory/background.md`
2. Choose your preferred language
3. Adapt `examples/himalaya/` to your region
4. Validate with observations

## 🤝 Contributing

We welcome contributions to **both** versions!

- **Bug fixes**: Submit PR to respective directory
- **Documentation**: Improve any README or docs
- **Examples**: Add new comparative examples
- **Features**: Propose enhancements

See `docs/contributing.md` for guidelines.

## 📚 Resources

### Original Publications
- Smith & Barstad (2004) - Linear theory of orographic precipitation
- Durran & Klemp (1982) - Moisture effects on buoyancy frequency

### Related Tools
- **Python ecosystem**: NumPy, SciPy, Matplotlib, Xarray
- **MATLAB ecosystem**: Signal Processing, Parallel Computing

### Community
- GitHub Issues: Bug reports and questions
- GitHub Discussions: General discussion

## 📝 Citation

When using either version:

```bibtex
@software{opi_model,
  title = {OPI: Orographic Precipitation and Isotopes},
  author = {Brandon, Mark and {Contributors}},
  year = {2024},
  note = {Available in MATLAB and Python}
}
```

## 🙏 Acknowledgments

- **Mark Brandon** (Yale University) - Original MATLAB implementation
- **AI Assistant & Contributors** - Python port and documentation

---

**Choose your path**: [MATLAB](matlab/) | [Python](python/) | [Compare](docs/comparison/)
