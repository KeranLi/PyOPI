# OPI Environment Setup Guide

This guide explains how to set up your conda environment for the OPI (Orographic Precipitation and Isotopes) Python package.

## Prerequisites

- Anaconda or Miniconda installed on your system
- Git (optional, for cloning the repository)

## Step-by-step Instructions

### 1. Open Anaconda Prompt or Terminal

On Windows: Open "Anaconda Prompt" from the Start Menu
On macOS/Linux: Open your terminal

### 2. Create a New Conda Environment

```bash
conda create -n opi-env python=3.9
```

### 3. Activate the Environment

```bash
conda activate opi-env
```

### 4. Navigate to Your Project Directory

```bash
cd path\to\OPI-Orographic-Precipitation-and-Isotopes
```

### 5. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify the Installation

```bash
python -c "import opi; print('OPI package imported successfully')"
```

### 7. Test Basic Functionality

```bash
python verify_installation.py
```

## Troubleshooting

### If you get a conda command not found error:
- Make sure Anaconda/Miniconda is installed
- Restart your terminal after installation
- On some systems, you may need to initialize conda: `conda init`

### If you get permission errors:
- On Windows: Run the command prompt as Administrator
- On macOS/Linux: Prepend the command with `sudo` (though this is not recommended)

### If packages fail to install:
- Update pip: `python -m pip install --upgrade pip`
- Try installing packages individually: `pip install numpy scipy matplotlib pandas xarray netcdf4 h5py`

## Using the Environment

Once your environment is set up:

1. Activate the environment: `conda activate opi-env`
2. Navigate to the project directory
3. Run your Python scripts
4. When finished, deactivate: `conda deactivate`

## Example Usage

```python
from opi.opi_calc_one_wind import opi_calc_one_wind

# Run the calculation
results = opi_calc_one_wind()
print(results)
```

## Additional Notes

- The environment `opi-env` is isolated, so packages installed here won't affect your base Python installation
- You can remove the environment later if needed: `conda env remove -n opi-env`
- To update dependencies: activate the environment and run `pip install -r requirements.txt` again