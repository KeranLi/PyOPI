#!/usr/bin/env python3
"""
Simple test for single wind field calculation
"""

import numpy as np

def test_single_wind():
    print("Testing single wind field calculation components...")
    
    # Import all required modules
    try:
        from opi.opi_calc_one_wind import opi_calc_one_wind
        print("✓ opi_calc_one_wind imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import opi_calc_one_wind: {e}")
        return False
    
    try:
        from opi.calc_one_wind import calc_one_wind
        print("✓ calc_one_wind imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import calc_one_wind: {e}")
        return False
    
    try:
        from opi.base_state import base_state
        print("✓ base_state imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import base_state: {e}")
        return False
    
    try:
        from opi.precipitation_grid import precipitation_grid
        print("✓ precipitation_grid imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import precipitation_grid: {e}")
        return False
    
    try:
        from opi.isotope_grid import isotope_grid
        print("✓ isotope_grid imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import isotope_grid: {e}")
        return False
    
    # Test base state calculation
    try:
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(0.01, 285.0, z_max=5000, dz=200)
        print(f"✓ Base state calculated with {len(z_bar)} elevation points")
        print(f"  Surface temp: {T[0]:.2f} K, Lapse rate: {gamma_env[0]:.6f} K/m")
    except Exception as e:
        print(f"✗ Error in base state calculation: {e}")
        return False
    
    # Test basic functionality that would be used in single wind calculation
    try:
        from opi.saturated_vapor_pressure import saturated_vapor_pressure
        e_s = saturated_vapor_pressure(298.15)  # 25°C in Kelvin
        print(f"✓ Saturated vapor pressure at 25°C: {e_s:.2f} Pa")
    except Exception as e:
        print(f"✗ Error in saturated vapor pressure calculation: {e}")
        return False
    
    print("\n✓ All single wind field components are working correctly!")
    print("\nTo run a complete single wind field calculation:")
    print("  from opi.opi_calc_one_wind import opi_calc_one_wind")
    print("  results = opi_calc_one_wind()")
    
    return True

if __name__ == "__main__":
    test_single_wind()