# OPI Examples

This directory contains example scripts demonstrating how to use OPI.

## Directory Structure

```
examples/
├── README.md                 # This file
├── simple_simulation.py      # Basic single-wind simulation example
└── data/                     # Example data files
    ├── gaussian_topo.mat
    └── ...
```

## Examples

### simple_simulation.py

Basic example demonstrating:
- Creating synthetic topography
- Running a single-wind OPI calculation
- Visualizing results with maps
- Saving output to file

```bash
cd OPI_python/examples
python simple_simulation.py
```

## Difference: Examples vs Scripts

### examples/ - Learning & Demonstration
- **Purpose:** Educational examples for learning OPI
- **Audience:** New users, students
- **Style:** Well-commented, step-by-step
- **Data:** Synthetic or small example datasets

### scripts/ - Production Workflows
- **Purpose:** Production-ready workflows for real research
- **Audience:** Researchers, advanced users
- **Style:** Efficient, configurable, command-line friendly
- **Data:** Real research datasets (e.g., Himalaya topography)

## Quick Start

```bash
# Run the simple example
python simple_simulation.py

# Output will be saved to output/ directory
```

## Creating Your Own Example

1. Copy `simple_simulation.py` as a template
2. Modify parameters (wind speed, direction, temperature, etc.)
3. Try different topography types
4. Experiment with visualization options

## API Reference

See `../README.md` for complete API documentation.

### Key Imports

```python
# Core calculation
from opi import calc_one_wind, calc_two_winds

# Visualization
from opi.viz.maps import plot_topography_map, plot_precipitation_map

# Tools
from opi.tools.synthetic_topography import create_synthetic_dem
```
