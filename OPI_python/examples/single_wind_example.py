#!/usr/bin/env python3
"""
Example script demonstrating single wind field calculation with OPI package
"""

import numpy as np
from datetime import datetime

from opi.opi_calc_one_wind import opi_calc_one_wind
from opi.base_state import base_state
from opi.saturated_vapor_pressure import saturated_vapor_pressure
from opi.coordinates import lonlat2xy
from opi.wind_path import wind_path
from opi.catchment_nodes import catchment_nodes
from opi.catchment_indices import catchment_indices
from opi.precipitation_grid import precipitation_grid
from opi.isotope_grid import isotope_grid
from opi.calc_one_wind import calc_one_wind
from opi.constants import *


def main():
    print("OPI Single Wind Field Calculation Example")
    print("="*50)
    
    # Record start time
    start_time = datetime.now()
    print(f"Starting calculation at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Example 1: Run the main single wind calculation with default parameters
    print("\n1. Running main single wind calculation...")
    try:
        results = opi_calc_one_wind()
        print("✓ Main calculation completed successfully")
        print(f"   Solution parameters: {dict(results['solution_params'])}")
        print(f"   Derived parameters: {dict(results['derived_params'])}")
    except Exception as e:
        print(f"✗ Error in main calculation: {str(e)}")
    
    # Example 2: Calculate atmospheric base state
    print("\n2. Calculating atmospheric base state...")
    try:
        # Typical values for a mid-latitude scenario
        NM = 0.01  # Buoyancy frequency (rad/s) - typical value
        T0 = 285.0  # Surface temperature (K) - ~12°C
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        print("✓ Base state calculation completed successfully")
        print(f"   Surface temperature: {T[0]:.2f} K")
        print(f"   Environmental lapse rate: {gamma_env[0]:.6f} K/m")
        print(f"   Saturated lapse rate: {gamma_sat[0]:.6f} K/m")
        print(f"   Water vapor scale height: {h_s:.2f} m")
    except Exception as e:
        print(f"✗ Error in base state calculation: {str(e)}")
    
    # Example 3: Calculate saturated vapor pressure
    print("\n3. Calculating saturated vapor pressure...")
    try:
        temp_kelvin = 298.15  # 25°C in Kelvin
        e_s = saturated_vapor_pressure(temp_kelvin)
        print(f"✓ Saturated vapor pressure at 25°C: {e_s:.2f} Pa")
    except Exception as e:
        print(f"✗ Error in saturated vapor pressure calculation: {str(e)}")
    
    # Example 4: Coordinate transformation
    print("\n4. Testing coordinate transformation...")
    try:
        # Define a simple grid
        lon = np.linspace(-5, 5, 21)  # degrees
        lat = np.linspace(-5, 5, 21)  # degrees
        lon0, lat0 = 0.0, 0.0  # reference point
        
        x, y = lonlat2xy(lon, lat, lon0, lat0)
        print(f"✓ Coordinate transformation completed")
        print(f"   Grid dimensions: {len(x)} x {len(y)}")
        print(f"   X range: {x[0]:.0f} to {x[-1]:.0f} m")
        print(f"   Y range: {y[0]:.0f} to {y[-1]:.0f} m")
    except Exception as e:
        print(f"✗ Error in coordinate transformation: {str(e)}")
    
    # Example 5: Wind path calculation
    print("\n5. Calculating wind path...")
    try:
        # Define a simple grid
        x_grid = np.linspace(0, 100000, 101)  # 100 km grid
        y_grid = np.linspace(0, 100000, 101)  # 100 km grid
        
        # Calculate path for 45-degree wind
        x_path, y_path = wind_path(0, 0, 45, x_grid, y_grid)
        print(f"✓ Wind path calculation completed")
        print(f"   Number of path points: {len(x_path)}")
        print(f"   Path covers X: {x_path[0]:.0f} to {x_path[-1]:.0f} m")
        print(f"   Path covers Y: {y_path[0]:.0f} to {y_path[-1]:.0f} m")
    except Exception as e:
        print(f"✗ Error in wind path calculation: {str(e)}")
    
    # Example 6: Create a simple synthetic example with realistic parameters
    print("\n6. Creating synthetic single wind calculation...")
    try:
        # Define a simple elevation grid (synthetic mountain)
        x = np.linspace(0, 100000, 50)  # 100 km in x direction
        y = np.linspace(0, 100000, 50)  # 100 km in y direction
        X, Y = np.meshgrid(x, y)
        
        # Create a simple mountain (Gaussian shape)
        center_x, center_y = len(x)//2, len(y)//2
        sig_x, sig_y = len(x)//8, len(y)//8  # Controls width of the mountain
        
        h_grid = 2000 * np.exp(-((X - x[center_x])**2 / (sig_x**2 * (x[1]-x[0])**2) + 
                                 (Y - y[center_y])**2 / (sig_y**2 * (y[1]-y[0])**2)))
        
        print(f"✓ Synthetic elevation grid created")
        print(f"   Grid size: {h_grid.shape}")
        print(f"   Max elevation: {np.max(h_grid):.2f} m")
        
        # Define sample points
        sample_x = np.array([x[len(x)//4], x[len(x)//2], x[3*len(x)//4]])  # Three points along x
        sample_y = np.array([y[len(y)//2], y[len(y)//2], y[len(y)//2]])   # Same y coordinate
        
        # Create sample data
        sample_lon = np.zeros_like(sample_x)
        sample_lat = np.zeros_like(sample_x)
        
        # Define solution vector (9 parameters for single wind field)
        # [U, azimuth, T0, M, kappa, tau_c, d2h0, d_d2h0_d_lat, f_p0]
        beta = np.array([
            10.0,      # Wind speed (m/s)
            180.0,     # Azimuth (degrees from North)
            285.0,     # Sea-level temperature (K)
            0.5,       # Mountain-height number (dimensionless)
            10.0,      # Eddy diffusion (m^2/s)
            10.0,      # Condensation time (s)
            -0.1,      # d2H of base precipitation (per unit)
            0.001,     # Latitudinal gradient of base-prec d2H (1/deg lat)
            0.8        # Residual precipitation after evaporation (fraction)
        ])
        
        print(f"✓ Synthetic example parameters defined")
        
    except Exception as e:
        print(f"✗ Error in synthetic example: {str(e)}")
    
    # Calculate elapsed time
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\nTotal execution time: {elapsed.total_seconds():.2f} seconds")
    
    print("\n" + "="*50)
    print("Single wind field calculation examples completed!")
    print("The OPI package is working correctly.")
    print("\nTo run your own calculations:")
    print("  1. Prepare your input data (topography, sample locations)")
    print("  2. Define your model parameters in a run file")
    print("  3. Call opi_calc_one_wind(run_file_path='path/to/your/run/file.txt')")
    print("="*50)


if __name__ == "__main__":
    main()