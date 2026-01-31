#!/usr/bin/env python3
"""
Simple verification script for OPI package installation
"""

def main():
    print("Verifying OPI package installation...")
    
    # Try importing the package
    try:
        import opi
        print("✓ OPI package imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OPI package: {e}")
        return False
    
    # Try importing specific modules
    try:
        from opi.constants import G, CPD, RD
        print("✓ Constants module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import constants: {e}")
        return False
    
    try:
        from opi.saturated_vapor_pressure import saturated_vapor_pressure
        print("✓ Saturated vapor pressure function imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import saturated_vapor_pressure: {e}")
        return False
    
    try:
        from opi.base_state import base_state
        print("✓ Base state function imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import base_state: {e}")
        return False
    
    try:
        from opi.opi_calc_one_wind import opi_calc_one_wind
        print("✓ Main calculation function imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import opi_calc_one_wind: {e}")
        return False
    
    # Test basic functionality
    try:
        # Test saturated vapor pressure calculation
        e_s = saturated_vapor_pressure(298.15)  # 25°C in Kelvin
        print(f"✓ Saturated vapor pressure at 25°C: {e_s:.2f} Pa")
    except Exception as e:
        print(f"✗ Error testing saturated vapor pressure: {e}")
        return False
    
    # Test coordinate conversion
    try:
        from opi.coordinates import lonlat2xy
        x, y = lonlat2xy([-120.0, -119.0], [40.0, 41.0], -120.0, 40.0)
        print(f"✓ Coordinate conversion successful: X range {x[0]:.0f}-{x[1]:.0f}m, Y range {y[0]:.0f}-{y[1]:.0f}m")
    except Exception as e:
        print(f"✗ Error testing coordinate conversion: {e}")
        return False
    
    print("\n✓ All tests passed! OPI package is ready to use.")
    print("\nTo run a calculation:")
    print("  from opi.opi_calc_one_wind import opi_calc_one_wind")
    print("  results = opi_calc_one_wind()")
    
    return True

if __name__ == "__main__":
    main()