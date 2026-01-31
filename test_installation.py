#!/usr/bin/env python3
"""
Test script to verify OPI package installation and basic functionality
"""

import sys
import numpy as np
from datetime import datetime

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ Failed to import NumPy")
        return False
    
    try:
        import scipy
        print("✓ SciPy imported successfully")
    except ImportError:
        print("✗ Failed to import SciPy")
        return False
    
    try:
        from opi.constants import G, CPD, RD
        print("✓ OPI constants imported successfully")
    except ImportError:
        print("✗ Failed to import OPI constants")
        return False
    
    try:
        from opi.saturated_vapor_pressure import saturated_vapor_pressure
        print("✓ Saturated vapor pressure function imported successfully")
    except ImportError:
        print("✗ Failed to import saturated vapor pressure function")
        return False
    
    try:
        from opi.base_state import base_state
        print("✓ Base state function imported successfully")
    except ImportError:
        print("✗ Failed to import base state function")
        return False
    
    try:
        from opi.coordinates import lonlat2xy, xy2lonlat
        print("✓ Coordinate functions imported successfully")
    except ImportError:
        print("✗ Failed to import coordinate functions")
        return False
    
    try:
        from opi.wind_path import wind_path
        print("✓ Wind path function imported successfully")
    except ImportError:
        print("✗ Failed to import wind path function")
        return False
    
    try:
        from opi.catchment_nodes import catchment_nodes
        print("✓ Catchment nodes function imported successfully")
    except ImportError:
        print("✗ Failed to import catchment nodes function")
        return False
    
    try:
        from opi.catchment_indices import catchment_indices
        print("✓ Catchment indices function imported successfully")
    except ImportError:
        print("✗ Failed to import catchment indices function")
        return False
    
    try:
        from opi.precipitation_grid import precipitation_grid
        print("✓ Precipitation grid function imported successfully")
    except ImportError:
        print("✗ Failed to import precipitation grid function")
        return False
    
    try:
        from opi.isotope_grid import isotope_grid
        print("✓ Isotope grid function imported successfully")
    except ImportError:
        print("✗ Failed to import isotope grid function")
        return False
    
    try:
        from opi.calc_one_wind import calc_one_wind
        print("✓ Calc one wind function imported successfully")
    except ImportError:
        print("✗ Failed to import calc one wind function")
        return False
    
    try:
        from opi.opi_calc_one_wind import opi_calc_one_wind
        print("✓ OPI calc one wind function imported successfully")
    except ImportError:
        print("✗ Failed to import OPI calc one wind function")
        return False
    
    return True


def test_basic_functions():
    """Test basic functionality of key functions"""
    print("\nTesting basic functionality...")
    
    # Test saturated vapor pressure
    try:
        from opi.saturated_vapor_pressure import saturated_vapor_pressure
        e_s = saturated_vapor_pressure(298.15)  # 25°C in Kelvin
        print(f"✓ Saturated vapor pressure at 25°C: {e_s:.2f} Pa")
    except Exception as e:
        print(f"✗ Error testing saturated vapor pressure: {str(e)}")
        return False
    
    # Test base state calculation with simple parameters
    try:
        from opi.base_state import base_state
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(0.01, 285.0, z_max=5000, dz=200)
        print(f"✓ Base state calculated with {len(z_bar)} elevation points")
        print(f"  Surface temp: {T[0]:.2f} K, Lapse rate: {gamma_env[0]:.6f} K/m")
    except Exception as e:
        print(f"✗ Error testing base state: {str(e)}")
        return False
    
    # Test coordinate conversions
    try:
        from opi.coordinates import lonlat2xy
        x, y = lonlat2xy([-120.0, -119.0], [40.0, 41.0], -120.0, 40.0)
        print(f"✓ Coordinate conversion: Lon/Lat to X/Y")
        print(f"  X range: {x[0]:.0f} to {x[1]:.0f} m")
        print(f"  Y range: {y[0]:.0f} to {y[1]:.0f} m")
    except Exception as e:
        print(f"✗ Error testing coordinate conversion: {str(e)}")
        return False
    
    # Test wind path calculation
    try:
        from opi.wind_path import wind_path
        import numpy as np
        x_grid = np.linspace(0, 100000, 101)  # 100 km grid
        y_grid = np.linspace(0, 100000, 101)  # 100 km grid
        x_path, y_path = wind_path(0, 0, 45, x_grid, y_grid)  # 45-degree wind
        print(f"✓ Wind path calculation with {len(x_path)} points")
    except Exception as e:
        print(f"✗ Error testing wind path: {str(e)}")
        return False
    
    return True


def main():
    """Main test function"""
    print("OPI Package Installation Test")
    print("="*50)
    
    start_time = datetime.now()
    
    # Test imports
    if not test_imports():
        print("\nInstallation test failed: Could not import required modules")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functions():
        print("\nInstallation test failed: Basic functionality errors")
        sys.exit(1)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n✓ All tests passed!")
    print(f"✓ OPI package is ready to use")
    print(f"✓ Test duration: {duration:.2f} seconds")
    
    print("\nTo run a calculation:")
    print("  from opi.opi_calc_one_wind import opi_calc_one_wind")
    print("  results = opi_calc_one_wind()")
    
    print("\nFor more information, see README_PYTHON.md")


if __name__ == "__main__":
    main()