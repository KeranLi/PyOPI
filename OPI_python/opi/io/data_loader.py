"""
Data Loading and Saving Utilities

Functions for loading topography, samples, and saving results in various formats
including NumPy arrays, NetCDF, GeoTIFF, and MATLAB files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import warnings


def load_topography_ascii(filepath: Union[str, Path]) -> Dict:
    """
    Load topography from ASCII grid file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to ASCII grid file (.asc)
        
    Returns
    -------
    dict : Dictionary with 'x', 'y', 'h_grid', and metadata
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        # Read header
        header = {}
        for _ in range(6):
            line = f.readline().strip()
            key, value = line.split()
            header[key.lower()] = float(value)
    
    ncols = int(header['ncols'])
    nrows = int(header['nrows'])
    xllcorner = header['xllcorner']
    yllcorner = header['yllcorner']
    cellsize = header['cellsize']
    
    # Read data
    h_grid = np.loadtxt(filepath, skiprows=6)
    
    # Handle nodata values
    if 'nodata_value' in header:
        nodata = header['nodata_value']
        h_grid = np.where(h_grid == nodata, np.nan, h_grid)
    
    # Create coordinate vectors
    x = xllcorner + np.arange(ncols) * cellsize
    y = yllcorner + np.arange(nrows) * cellsize
    
    return {
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'cellsize': cellsize,
        'xllcorner': xllcorner,
        'yllcorner': yllcorner
    }


def load_topography_netcdf(filepath: Union[str, Path], 
                           x_var: str = 'x',
                           y_var: str = 'y', 
                           z_var: str = 'z') -> Dict:
    """
    Load topography from NetCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to NetCDF file
    x_var, y_var, z_var : str
        Variable names for x, y coordinates and elevation
        
    Returns
    -------
    dict : Dictionary with 'x', 'y', 'h_grid'
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for NetCDF support. Install with: pip install xarray netCDF4")
    
    ds = xr.open_dataset(filepath)
    
    x = ds[x_var].values
    y = ds[y_var].values
    h_grid = ds[z_var].values
    
    ds.close()
    
    return {
        'x': x,
        'y': y,
        'h_grid': h_grid
    }


def load_topography_geotiff(filepath: Union[str, Path]) -> Dict:
    """
    Load topography from GeoTIFF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to GeoTIFF file
        
    Returns
    -------
    dict : Dictionary with 'x', 'y', 'h_grid' and georeferencing info
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF support. Install with: pip install rasterio")
    
    with rasterio.open(filepath) as src:
        h_grid = src.read(1)
        
        # Get coordinate vectors
        rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        x = np.array(xs[0, :])
        y = np.array(ys[:, 0])
        
        result = {
            'x': x,
            'y': y,
            'h_grid': h_grid,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds
        }
    
    return result


def load_topography(filepath: Union[str, Path]) -> Dict:
    """
    Load topography from file (auto-detect format).
    
    Supports: .asc (ASCII), .nc/.nc4 (NetCDF), .tif/.tiff (GeoTIFF), .npy (NumPy)
    
    Parameters
    ----------
    filepath : str or Path
        Path to topography file
        
    Returns
    -------
    dict : Dictionary with 'x', 'y', 'h_grid'
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.asc':
        return load_topography_ascii(filepath)
    elif suffix in ['.nc', '.nc4', '.netcdf']:
        return load_topography_netcdf(filepath)
    elif suffix in ['.tif', '.tiff']:
        return load_topography_geotiff(filepath)
    elif suffix == '.npy':
        data = np.load(filepath, allow_pickle=True).item()
        if not all(k in data for k in ['x', 'y', 'h_grid']):
            raise ValueError("NumPy file must contain 'x', 'y', 'h_grid'")
        return data
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_topography_netcdf(topography: Dict, filepath: Union[str, Path],
                           x_var: str = 'x', y_var: str = 'y', z_var: str = 'elevation'):
    """
    Save topography to NetCDF file.
    
    Parameters
    ----------
    topography : dict
        Dictionary with 'x', 'y', 'h_grid'
    filepath : str or Path
        Output file path
    x_var, y_var, z_var : str
        Variable names
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for NetCDF support")
    
    ds = xr.Dataset(
        {
            z_var: ([y_var, x_var], topography['h_grid'])
        },
        coords={
            x_var: topography['x'],
            y_var: topography['y']
        }
    )
    
    ds[z_var].attrs['units'] = 'meters'
    ds[z_var].attrs['long_name'] = 'Surface elevation'
    
    ds.to_netcdf(filepath)
    ds.close()


def save_topography_geotiff(topography: Dict, filepath: Union[str, Path],
                            crs: Optional[str] = None):
    """
    Save topography to GeoTIFF file.
    
    Parameters
    ----------
    topography : dict
        Dictionary with 'x', 'y', 'h_grid'
    filepath : str or Path
        Output file path
    crs : str, optional
        Coordinate reference system (e.g., 'EPSG:4326')
    """
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise ImportError("rasterio is required for GeoTIFF support")
    
    h_grid = topography['h_grid']
    x = topography['x']
    y = topography['y']
    
    # Calculate transform
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    transform = from_origin(x[0], y[-1], dx, abs(dy))
    
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=h_grid.shape[0],
        width=h_grid.shape[1],
        count=1,
        dtype=h_grid.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(h_grid, 1)


def load_samples_csv(filepath: Union[str, Path], 
                     lon_col: str = 'longitude',
                     lat_col: str = 'latitude',
                     d2h_col: str = 'd2H',
                     d18o_col: str = 'd18O') -> Dict:
    """
    Load sample data from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    lon_col, lat_col : str
        Column names for coordinates
    d2h_col, d18o_col : str
        Column names for isotope values
        
    Returns
    -------
    dict : Dictionary with sample data
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for CSV support")
    
    df = pd.read_csv(filepath)
    
    samples = {
        'lon': df[lon_col].values,
        'lat': df[lat_col].values,
        'd2H': df[d2h_col].values / 1000,  # Convert ‰ to fraction
        'd18O': df[d18o_col].values / 1000,
    }
    
    # Optional columns
    if 'type' in df.columns:
        samples['type'] = df['type'].values
    if 'elevation' in df.columns:
        samples['elevation'] = df['elevation'].values
    
    return samples


def save_results(result, filepath: Union[str, Path], format: str = 'npz'):
    """
    Save OPI results to file.
    
    Parameters
    ----------
    result : object
        OPI result object (OneWindResult, TwoWindsResult, etc.)
    filepath : str or Path
        Output file path
    format : str
        Output format: 'npz' (NumPy), 'nc' (NetCDF), 'mat' (MATLAB)
    """
    filepath = Path(filepath)
    
    # Extract data from result object
    data = {}
    for attr in dir(result):
        if not attr.startswith('_'):
            value = getattr(result, attr)
            if isinstance(value, np.ndarray):
                data[attr] = value
            elif isinstance(value, (int, float)):
                data[attr] = value
    
    if format == 'npz':
        np.savez(filepath, **data)
    elif format == 'nc':
        save_results_netcdf(data, filepath)
    elif format == 'mat':
        save_results_matlab(data, filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_results_netcdf(data: Dict, filepath: Union[str, Path]):
    """Save results to NetCDF format."""
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray is required for NetCDF support")
    
    # Separate scalars and arrays
    scalars = {k: v for k, v in data.items() if np.isscalar(v)}
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    
    # Create dataset
    ds = xr.Dataset()
    
    # Add scalars as global attributes
    ds.attrs.update(scalars)
    
    # Add arrays as variables
    for name, arr in arrays.items():
        if arr.ndim == 2:
            ds[name] = (['y', 'x'], arr)
        elif arr.ndim == 1:
            ds[name] = (['index'], arr)
    
    ds.to_netcdf(filepath)
    ds.close()


def save_results_matlab(data: Dict, filepath: Union[str, Path]):
    """Save results to MATLAB format."""
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError("scipy is required for MATLAB support")
    
    savemat(filepath, data)


def load_results(filepath: Union[str, Path]) -> Dict:
    """
    Load OPI results from file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to results file (.npz, .nc, or .mat)
        
    Returns
    -------
    dict : Dictionary with result data
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.npz':
        return dict(np.load(filepath, allow_pickle=True))
    elif suffix in ['.nc', '.nc4']:
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required for NetCDF support")
        ds = xr.open_dataset(filepath)
        result = {name: ds[name].values for name in ds.data_vars}
        result.update(ds.attrs)
        ds.close()
        return result
    elif suffix == '.mat':
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required for MATLAB support")
        return loadmat(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
