# OPI Python Project Structure

This document describes the organized project structure.

## 📁 Directory Overview

```
OPI_python/
├── README.md                   # Main project README
├── PROJECT_STRUCTURE.md        # This file
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── setup_environment.bat       # Windows setup script
├── setup_environment.sh        # Unix setup script
│
├── docs/                       # 📚 Documentation
│   ├── README.md              # Documentation index
│   ├── ENV_SETUP_GUIDE.md     # Environment setup
│   ├── README_PYTHON.md       # Original Python README
│   │
│   ├── Planning/              # 📋 Project Planning
│   │   ├── FUNCTIONALITY_PLAN.md
│   │   ├── FUNCTIONALITY_GAP_ANALYSIS.md
│   │   └── IMPLEMENTATION_PLAN.md
│   │
│   ├── Reports/               # 📊 Progress Reports
│   │   ├── PROGRESS_REPORT.md
│   │   ├── COMPLETION_REPORT.md
│   │   └── FINAL_COMPLETION_REPORT.md
│   │
│   └── Refactoring/           # 🏗️ Architecture
│       ├── REFACTORING_PLAN.md
│       ├── REFACTORING_DEMO.md
│       └── REFACTORING_SUMMARY.md
│
├── opi/                        # 📦 Main Package
│   ├── __init__.py            # Package exports
│   ├── __main__.py            # CLI entry point
│   │
│   ├── models/                # 📊 Data Models (NEW)
│   │   ├── domain.py          # Grid, Topography
│   │   ├── parameters.py      # Parameter classes
│   │   ├── results.py         # Result classes
│   │   └── config.py          # Configuration
│   │
│   ├── core/                  # 🔧 Core Abstractions (NEW)
│   │   ├── base.py            # Base classes
│   │   ├── calculator.py      # Calculator interface
│   │   ├── optimizer.py       # Optimizer interface
│   │   ├── interfaces.py      # Protocols
│   │   └── exceptions.py      # Exception hierarchy
│   │
│   ├── solvers/               # ⚛️ Physical Solvers (NEW)
│   │   ├── fourier.py         # FFT solver
│   │   ├── precipitation.py   # LTOP precipitation
│   │   └── isotope.py         # Isotope calculator
│   │
│   ├── calculators/           # 🧮 Application Layer (NEW)
│   │   └── one_wind_new.py    # Refactored calculator
│   │
│   ├── io/                    # 📥 Input/Output
│   ├── infrastructure/        # 🏭 Infrastructure
│   ├── plugins/               # 🔌 Plugin System
│   │
│   └── [Legacy Modules]       # 📁 Original Implementation
│       ├── base_state.py
│       ├── calc_one_wind.py
│       ├── fourier_solution.py
│       ├── precipitation_grid.py
│       ├── isotope_grid.py
│       ├── opi_calc_one_wind.py
│       ├── opi_calc_two_winds.py
│       ├── opi_fit_one_wind.py
│       ├── opi_fit_two_winds.py
│       └── ...
│
├── examples/                   # 📝 Example Scripts
│   ├── comprehensive_example.py
│   ├── complete_workflow_example.py
│   ├── single_wind_example.py
│   ├── process_example_data.py
│   ├── reproduce_original_results.py
│   ├── reproduce_with_real_data.py
│   ├── test_single_wind.py
│   └── verify_installation.py
│
├── tests/                      # 🧪 Tests
│   ├── extract_test_data.m    # MATLAB test data extraction
│   ├── matlab_reference_data/ # Reference data
│   ├── test_installation.py
│   ├── test_plotting.py
│   └── verify_installation.py
│
├── scripts/                    # 🛠️ Utility Scripts
│   ├── functionality_comparison.py
│   ├── setup_environment.bat
│   └── setup_environment.sh
│
└── [Build Artifacts]
    ├── opi_orographic_precipitation_isotopes.egg-info/
    └── __pycache__/
```

## 📂 Directory Descriptions

### `/docs` - Documentation
All project documentation organized by category:
- **Planning**: Initial planning documents
- **Reports**: Progress and completion reports
- **Refactoring**: Architecture refactoring documents

### `/opi` - Main Package
The core Python package with two architectures:

#### Legacy Architecture (Working)
Original MATLAB translation:
- `opi_calc_*.py` - Main calculation functions
- `*_grid.py` - Grid calculation modules
- `calc_one_wind.py` - Core calculation logic

#### New Architecture (Refactored)
Pythonic object-oriented design:
- `models/` - Data classes with type hints
- `core/` - Abstract base classes
- `solvers/` - Physical algorithm implementations
- `calculators/` - High-level calculator classes

### `/examples` - Examples
Working example scripts demonstrating usage:
- Basic calculations
- Complete workflows
- Real data processing

### `/tests` - Tests
Test scripts and reference data:
- Installation tests
- Verification scripts
- MATLAB reference data

### `/scripts` - Utility Scripts
Helper scripts for setup and comparison.

## 🔄 File Organization Principles

### 1. Separation of Concerns
- **Code**: `/opi` directory
- **Documentation**: `/docs` directory
- **Examples**: `/examples` directory
- **Tests**: `/tests` directory

### 2. Dual Architecture
- **Legacy**: Root of `/opi` - working MATLAB translation
- **New**: Subdirectories in `/opi` - refactored architecture

### 3. Documentation Hierarchy
- **Main README**: Project root
- **Detailed Docs**: `/docs` with categorization
- **Doc Index**: `/docs/README.md`

## 🎯 Key Files

| File | Location | Purpose |
|:-----|:---------|:--------|
| `README.md` | Root | Project overview and quick start |
| `requirements.txt` | Root | Python dependencies |
| `setup.py` | Root | Package installation |
| `__main__.py` | `opi/` | CLI entry point |
| `docs/README.md` | `docs/` | Documentation index |

## 🚀 Usage

```bash
# Install package
pip install -e .

# Run CLI
python -m opi info

# Run examples
cd examples
python comprehensive_example.py

# Run tests
python -m opi test
```

## 📊 Statistics

| Category | Count |
|:---------|:------|
| Python Modules | 30+ |
| Documentation Files | 11 |
| Example Scripts | 8 |
| Test Scripts | 4 |
| Architecture Layers | 4 |
