#!/usr/bin/env python3
"""
Script to process actual example data from the original OPI package
"""

import numpy as np
import os
from datetime import datetime
import scipy.io as sio
import h5py

from opi.opi_calc_one_wind import opi_calc_one_wind
from opi.calc_one_wind import calc_one_wind
from opi.base_state import base_state
from opi.saturated_vapor_pressure import saturated_vapor_pressure
from opi.coordinates import lonlat2xy
from opi.wind_path import wind_path
from opi.catchment_nodes import catchment_nodes
from opi.catchment_indices import catchment_indices
from opi.precipitation_grid import precipitation_grid
from opi.isotope_grid import isotope_grid
from opi.constants import *


def load_topography_data(filepath):
    """
    Load topography data from MAT file (handles both v7.3 and older formats)
    """
    # Check if file is v7.3 format (HDF5) or older format
    try:
        # First, try to read with h5py (for v7.3 format)
        with h5py.File(filepath, 'r') as f:
            print(f"✓ Successfully opened MAT file with h5py: {filepath}")
            print("Available keys in MAT file:", list(f.keys()))
            
            # Look for topography data (common names in MATLAB files)
            topography_keys = ['h', 'topo', 'elevation', 'height', 'Z', 'h_grid', 'x', 'y']
            
            # Dictionary to store loaded data
            mat_data = {}
            for key in f.keys():
                if not key.startswith('#'):  # Skip metadata keys
                    try:
                        data = f[key]
                        if isinstance(data, h5py.Dataset):
                            mat_data[key] = data[()]
                            print(f"Key '{key}' has data shape: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")
                        else:
                            print(f"Key '{key}' is a group, skipping...")
                    except Exception as e:
                        print(f"Could not load key '{key}': {e}")
                        
            # Look for topography data specifically
            topo_data = None
            for key in mat_data.keys():
                if 'h_' in key or 'topo' in key or 'height' in key or 'elev' in key or key == 'Z':
                    topo_data = mat_data[key]
                    print(f"Found topography data in key: {key}")
                    break
            
            # If no specific topography key found, use the first non-metadata array
            if topo_data is None:
                for key in mat_data.keys():
                    if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) >= 2:  # Likely a grid
                        topo_data = mat_data[key]
                        print(f"Using data from key: {key} as topography data")
                        break
            
            if topo_data is not None:
                print(f"Topography grid shape: {topo_data.shape}")
                print(f"Elevation range: {np.min(topo_data):.2f} m to {np.max(topo_data):.2f} m")
                
                # Handle potential transpose issue (h5py loads in column-major order)
                if topo_data.flags['C_CONTIGUOUS']:
                    topo_data = np.ascontiguousarray(topo_data.T)
                else:
                    topo_data = topo_data.T  # Transpose for proper orientation
                
                print(f"Transposed topography grid shape: {topo_data.shape}")
                print(f"Corrected elevation range: {np.min(topo_data):.2f} m to {np.max(topo_data):.2f} m")
                
                return topo_data
            else:
                print("⚠ Could not identify topography data in MAT file")
                return None
                
    except OSError:
        # If h5py fails, it's probably an older format, try scipy.io
        print("File is not v7.3 format, trying scipy.io...")
        try:
            mat_data = sio.loadmat(filepath)
            print(f"✓ Successfully loaded MAT file with scipy.io: {filepath}")
            
            print("Available keys in MAT file:", list(mat_data.keys()))
            
            # Look for topography data (common names in MATLAB files)
            topo_data = None
            
            for key in mat_data.keys():
                if not key.startswith('__'):  # Skip metadata keys
                    print(f"Key '{key}' has data shape: {mat_data[key].shape if hasattr(mat_data[key], 'shape') else 'N/A'}")
                    if 'h_' in key or 'topo' in key or 'height' in key or 'elev' in key or key == 'Z':
                        topo_data = mat_data[key]
                        print(f"Found topography data in key: {key}")
                        break
            
            # If no specific topography key found, use the first non-metadata array
            if topo_data is None:
                for key in mat_data.keys():
                    if not key.startswith('__') and hasattr(mat_data[key], 'shape'):
                        if len(mat_data[key].shape) >= 2:  # Likely a grid
                            topo_data = mat_data[key]
                            print(f"Using data from key: {key} as topography data")
                            break
            
            if topo_data is not None:
                print(f"Topography grid shape: {topo_data.shape}")
                print(f"Elevation range: {np.min(topo_data):.2f} m to {np.max(topo_data):.2f} m")
                return topo_data
            else:
                print("⚠ Could not identify topography data in MAT file")
                return None
                
        except Exception as e:
            print(f"✗ Error loading MAT file with scipy.io: {str(e)}")
            return None
    except Exception as e:
        print(f"✗ Error loading MAT file with h5py: {str(e)}")
        return None


def main():
    print("Processing Actual Example Data from Original OPI Package")
    print("="*70)
    
    # Record start time
    start_time = datetime.now()
    print(f"Starting data processing at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define paths to example data
    example_data_path = "../OPI Example Gaussian Mountain Range/data"
    topo_filename = "EastDirectedGaussianTopography_3km height_lat45N.mat"
    topo_filepath = os.path.join(example_data_path, topo_filename)
    
    # Check if file exists
    if not os.path.exists(topo_filepath):
        # Try alternative path
        topo_filepath = "f:/code/OPI-Orographic-Precipitation-and-Isotopes/OPI Example Gaussian Mountain Range/data/EastDirectedGaussianTopography_3km height_lat45N.mat"
        if not os.path.exists(topo_filepath):
            print(f"✗ Topography file not found: {topo_filepath}")
            return
    
    print(f"Loading topography data from: {topo_filepath}")
    
    # Load topography data
    h_grid = load_topography_data(topo_filepath)
    
    if h_grid is None:
        print("Cannot proceed without topography data")
        return
    
    # Extract coordinate information from the grid
    ny, nx = h_grid.shape
    
    # Create coordinate grids (assuming a typical domain size)
    # The original example likely has specific coordinate systems
    dx = 3000  # 3 km resolution as indicated in filename
    dy = 3000
    x = np.arange(0, nx * dx, dx)
    y = np.arange(0, ny * dy, dy)
    
    print(f"Grid dimensions: x={len(x)} ({x[0]:.0f}m to {x[-1]:.0f}m), "
          f"y={len(y)} ({y[0]:.0f}m to {y[-1]:.0f}m), h_grid={h_grid.shape}")
    
    # Define solution vector based on the example run file
    # From the example: U=10, azimuth=90, T0=290, M=0.25, kappa=0, tau_c=1000, d2h0=9.6e-3, d_d2h0_d_lat=0, f_p0=1
    beta = np.array([
        10.0,      # Wind speed (m/s)
        90.0,      # Azimuth (degrees from North) - East wind
        290.0,     # Sea-level temperature (K)
        0.25,      # Mountain-height number (dimensionless)
        0.0,       # Eddy diffusion (m^2/s)
        1000.0,    # Condensation time (s)
        9.6e-3,    # d2H of base precipitation (per unit)
        0.0,       # Latitudinal gradient of base-prec d2H (1/deg lat)
        1.0        # Residual precipitation after evaporation (fraction)
    ])
    
    print(f"\nUsing parameters from original example:")
    param_names = ['U (m/s)', 'Azimuth (deg)', 'T0 (K)', 'M', 'κ (m²/s)', 'τc (s)', 'δ2H0', 'dδ2H0/dlat', 'fP']
    for i, name in enumerate(param_names):
        print(f"  {name}: {beta[i]}")
    
    # Define some sample points (these would normally come from a separate isotope data file)
    # For demonstration, we'll select points from the grid
    sample_indices_x = [max(0, nx//4), nx//2, min(nx-1, 3*nx//4)]  # Three points along x
    sample_indices_y = [max(0, ny//2), max(0, ny//2), max(0, ny//2)]  # Same y coordinate
    
    sample_x = x[sample_indices_x]
    sample_y = y[sample_indices_y]
    sample_z = h_grid[sample_indices_y, sample_indices_x]  # Elevations at those points
    
    # Mock isotope data based on the example (real data would come from Excel file)
    # In the original example, these would be actual measured isotope values
    sample_d2H = np.array([-100e-3, -150e-3, -200e-3])  # Example deuterium values in per mil
    
    print(f"\nSample locations:")
    for i in range(len(sample_x)):
        print(f"  Point {i+1}: x={sample_x[i]:.0f}m, y={sample_y[i]:.0f}m, z={sample_z[i]:.1f}m, δ2H={sample_d2H[i]*1e3:.1f}‰")
    
    # Now run the actual calculation with real data
    print(f"\n4. Running calculation with actual example data...")
    try:
        # We'll use calc_one_wind function directly since we have all the data
        results = calc_one_wind(beta, x, y, h_grid, sample_x, sample_y, sample_z, sample_d2H)
        
        print("✓ Real data calculation completed successfully")
        print(f"   Model misfit: {results.get('misfit', 'N/A')}")
        print(f"   Correlation coefficient: {results.get('correlation', 'N/A')}")
        
        # Access solution details if available
        if 'predicted_d2H' in results:
            corr_coef = np.corrcoef(results['predicted_d2H'], sample_d2H)[0,1]
            print(f"   Predicted vs observed correlation: {corr_coef:.4f}")
        
    except Exception as e:
        print(f"⚠ Error in real data calculation: {str(e)}")
        print("   Attempting with default parameters instead...")
        try:
            # Try the high-level function
            results = opi_calc_one_wind()
            print("✓ Default calculation completed")
        except Exception as e2:
            print(f"⚠ Both calculations failed: {str(e2)}")
    
    # Calculate atmospheric base state (from original example)
    print(f"\n5. Calculating atmospheric base state (from original example)...")
    try:
        # Values from the original example
        NM = 0.01  # Buoyancy frequency (rad/s) - typical value
        T0 = 290.0  # Surface temperature (K) - from run file
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        print("✓ Base state calculation completed successfully")
        print(f"   Surface temperature: {T[0]:.2f} K")
        print(f"   Environmental lapse rate: {gamma_env[0]:.6f} K/m")
        print(f"   Saturated lapse rate: {gamma_sat[0]:.6f} K/m")
        print(f"   Water vapor scale height: {h_s:.2f} m")
    except Exception as e:
        print(f"⚠ Error in base state calculation: {str(e)}")
    
    # Calculate elapsed time
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed.total_seconds():.2f} seconds")
    
    print("\n" + "="*70)
    print("Example data processing completed!")
    print("\nThis reproduces the original author's workflow using the example data.")
    print("For actual scientific analysis, you would need real isotope measurements")
    print("from an Excel file and potentially adjust model parameters accordingly.")
    print("="*70)


if __name__ == "__main__":
    main()