# OPI Python - Root Directory Guide

## Directory Structure

```
OPI_python/                     # Project root
│
├── opi/                        # Main Python package
│   ├── core/                   # Core computations (9 modules)
│   ├── models/                 # Model interfaces (2 modules)
│   ├── solvers/                # Optimization (1 module)
│   ├── io/                     # I/O utilities (2 modules)
│   ├── utils/                  # Utilities (3 modules)
│   ├── visualization/          # Visualization (4 modules)
│   ├── parallel/               # Parallel processing (2 modules)
│   └── constants.py            # Physical constants
│
├── tests/                      # Test suite (87+ tests)
│   ├── core/
│   ├── models/
│   └── solvers/
│
├── docs/                       # Documentation
│   ├── guides/                 # Detailed guides
│   ├── tutorials/              # Step-by-step tutorials
│   ├── api/                    # API reference
│   └── examples/               # Example docs
│
├── examples/                   # Example scripts
│   ├── simple_example.py
│   ├── comprehensive_example.py
│   └── legacy/                 # Old examples (for reference)
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_quick_start.ipynb
│   └── ...
│
├── scripts/                    # Utility scripts
│   ├── verify_structure.py     # Verify package structure
│   └── cleanup_old_files.py    # Cleanup old files
│
├── README.md                   # Main project documentation
├── LICENSE                     # MIT License
├── setup.py                    # Package installation
└── ROOT_DIRECTORY.md           # This file
```

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation - **Start here** |
| `LICENSE` | MIT License |
| `setup.py` | Package installation configuration |
| `ROOT_DIRECTORY.md` | This file - root directory guide |

## Documentation

| Location | Content |
|----------|---------|
| `README.md` | Project overview, quick start, features |
| `docs/guides/` | Detailed guides (architecture, structure) |
| `docs/tutorials/` | Step-by-step tutorials |
| `docs/api/` | API reference documentation |
| `examples/` | Working example scripts |
| `notebooks/` | Jupyter notebooks |

## Quick Start

1. **Installation**: See `README.md` or `docs/installation.rst`
2. **Tutorials**: See `docs/tutorials/`
3. **Examples**: See `examples/`
4. **API Reference**: See `docs/api/`

## For Developers

1. Check `docs/guides/PROJECT_STRUCTURE.md` for package architecture
2. Run `python scripts/verify_structure.py` to verify setup
3. Run `pytest tests/` to run the test suite

## File Organization

- **Package code**: `opi/` (23 modules)
- **Tests**: `tests/` (87+ tests)
- **Documentation**: `docs/` (Sphinx docs)
- **Examples**: `examples/` (working examples)
- **Utilities**: `scripts/` (maintenance scripts)
