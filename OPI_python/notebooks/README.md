# OPI Jupyter Notebooks

Interactive tutorials for learning and using OPI Python.

## Quick Start

```bash
cd OPI_python/notebooks
jupyter notebook
```

## Notebooks

### 1. [01_quick_start.ipynb](01_quick_start.ipynb) - Quick Start Guide
**Level:** Beginner | **Time:** ~15 minutes

Learn the basics of OPI:
- Create synthetic topography
- Run a simple simulation
- Visualize results
- Save output

**Key Topics:**
- `opi.tools.synthetic_topography.create_synthetic_dem()`
- `opi.calc_one_wind()` (low-level calculation)
- `opi.viz.maps.plot_*_map()` for visualization

### 2. [02_parameter_fitting.ipynb](02_parameter_fitting.ipynb) - Parameter Fitting
**Level:** Intermediate | **Time:** ~20 minutes

Learn parameter optimization:
- Create synthetic sample data
- Set up a run file for fitting
- Run CRS3 optimization
- Analyze fitting results

**Key Topics:**
- `opi.app.opi_fit_one_wind()`
- `opi.io.run_file.parse_run_file()`
- Residual analysis
- Sample comparison plots

### 3. [03_data_formats.ipynb](03_data_formats.ipynb) - Data Formats
**Level:** Reference | **Time:** ~10 minutes

Complete guide to data input formats:
- MATLAB .mat files
- Excel and CSV
- NetCDF (climate data)
- GeoTIFF (GIS data)
- JSON (configuration)

**Key Topics:**
- `opi.io.data_loader.grid_read()`
- `opi.io.data_loader.get_input()`
- `pd.read_excel()` / `pd.read_csv()`
- `xarray` for NetCDF
- `rasterio` for GeoTIFF

### 4. [04_reproduce_gaussian_example.ipynb](04_reproduce_gaussian_example.ipynb) - Gaussian Example
**Level:** Intermediate | **Time:** ~20 minutes

Reproduce the classic Gaussian mountain example:
- Load example topography
- Run single wind calculation
- Compare with published results
- Generate publication-quality figures

**Key Topics:**
- Loading .mat topography files
- Using `opi.app.opi_calc_one_wind()`
- Advanced visualization with `opi.viz.advanced`

## Module Structure (v1.0.0+)

```python
# Core calculation (low-level)
from opi import calc_one_wind, calc_two_winds

# High-level application functions
from opi.app import opi_calc_one_wind, opi_calc_two_winds
from opi.app import opi_fit_one_wind, opi_fit_two_winds

# Visualization
from opi.viz.maps import plot_topography_map, plot_precipitation_map
from opi.viz.plots import plot_sample_comparison, plot_residuals

# I/O utilities
from opi.io.data_loader import grid_read, get_input
from opi.io.run_file import parse_run_file, write_run_file

# Tools
from opi.tools.synthetic_topography import create_synthetic_dem
```

## Running Notebooks

### Option 1: Local Jupyter

```bash
# Install jupyter
pip install jupyter

# Start server
cd OPI_python/notebooks
jupyter notebook

# Or use JupyterLab
jupyter lab
```

### Option 2: VS Code

1. Install VS Code Python extension
2. Open `.ipynb` file
3. Select Python interpreter
4. Run cells with Shift+Enter

### Option 3: Google Colab

Upload notebooks to Google Colab:
```python
# In Colab, install OPI
!git clone https://github.com/your-repo/OPI.git
!cd OPI/OPI_python && pip install -e .
```

## Tips for Using Notebooks

1. **Run cells in order** - Each notebook builds on previous cells
2. **Modify parameters** - Try different wind speeds, temperatures, etc.
3. **Experiment** - Use synthetic data to explore model behavior
4. **Save your work** - Use "File > Make a Copy" to save modified notebooks

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'opi'`

**Solution:** Add parent directory to path
```python
import sys
sys.path.insert(0, '..')  # or '/path/to/OPI_python'
import opi
```

### Issue: Plots not showing

**Solution:** Enable matplotlib inline
```python
%matplotlib inline
# or for interactive plots:
%matplotlib widget
```

### Issue: Out of memory

**Solution:** Reduce grid resolution
```python
from opi.tools.synthetic_topography import create_synthetic_dem

# Use coarser grid
dem = create_synthetic_dem(
    grid_size=(300e3, 300e3),
    grid_spacing=(5000, 5000)  # 5km instead of 2km
)
```

## Migration from Old API

If you have code using the old API:

```python
# OLD (pre-v1.0)
from opi import opi_calc_one_wind
result = opi_calc_one_wind(run_file)

# NEW (v1.0+)
from opi.app import opi_calc_one_wind
result = opi_calc_one_wind(run_file)
```

```python
# OLD
from opi import create_synthetic_dem

# NEW
from opi.tools.synthetic_topography import create_synthetic_dem
```

## Next Steps

After completing the notebooks:
- See `../scripts/` for production workflows
- Read `../README.md` for API reference
- Check `../docs/MATLAB_PYTHON_COMPARISON.md` for MATLAB migration
