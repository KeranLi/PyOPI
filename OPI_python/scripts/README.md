# OPI Scripts

Production-ready workflow scripts for running OPI simulations with real data.

## Available Scripts

### run033_replica.py

**Purpose:** 1:1 replica of MATLAB OPI 3.7 run033 simulation

**Features:**
- Real Himalaya topography (GEBCO 1km)
- Real water isotope samples (Himalaya dataset)
- Two-wind model with continental divide splitting
- Full MATLAB compatibility

**Usage:**
```bash
# Ensure data files are in place:
# - OPI_python/data/Himalaya topography_April 2023_Gebco1kmMakima.mat
# - OPI_python/data/Himalaya Water Isotopes 8 March 2023.xlsx

cd OPI_python/scripts
python run033_replica.py
```

**Output:**
- Results saved to `../outputs/run033_replica_YYYYMMDD_HHMMSS.npz`
- Contains: grids, parameters, predictions, statistics

## Directory Structure

```
scripts/
├── README.md              # This file
└── run033_replica.py      # MATLAB run033 replica
```

## Difference: Scripts vs Examples

### scripts/ - Production Workflows
- **Purpose:** Run real research simulations
- **Audience:** Researchers using actual datasets
- **Style:** Command-line friendly, configurable paths
- **Data:** Large real-world datasets (Himalaya, etc.)

### examples/ - Learning & Testing
- **Purpose:** Learn OPI basics
- **Audience:** New users
- **Style:** Step-by-step, heavily commented
- **Data:** Synthetic or small example data

## Adding New Scripts

When adding a new production script:

1. Use the new modular API:
```python
from opi import calc_one_wind, calc_two_winds
from opi.app import opi_calc_one_wind, opi_calc_two_winds
```

2. Support configurable data paths
3. Save results with timestamps
4. Add entry to this README

## Data Requirements

Scripts in this directory expect real data files:

- Topography: `.mat` files (MATLAB v7.3 format supported)
- Samples: `.xlsx` or `.csv` files
- Continental divide: `.mat` polyline files

Place data files in `../data/` directory.
