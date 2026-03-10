"""
Test core physics modules against expected behavior

This script tests the core physics calculations to ensure they match
the MATLAB implementation.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))


def test_saturated_vapor_pressure():
    """Test saturated vapor pressure calculation"""
    print("Testing saturated_vapor_pressure...")
    from opi.saturated_vapor_pressure import saturated_vapor_pressure
    
    # Test at key temperatures
    T_test = np.array([273.15, 268.0, 258.0, 248.0])
    e_s = saturated_vapor_pressure(T_test)
    
    # Check values are reasonable (should be around 600 Pa at 0°C)
    assert e_s[0] > 500 and e_s[0] < 700, f"e_s at 0°C should be ~610 Pa, got {e_s[0]}"
    assert e_s[1] < e_s[0], "e_s should decrease with temperature"
    
    print(f"  At 0C: {e_s[0]:.1f} Pa (expected ~610 Pa)")
    print(f"  At -5C: {e_s[1]:.1f} Pa")
    print("  PASSED")


def test_fractionation():
    """Test isotope fractionation calculations"""
    print("\nTesting fractionation calculations...")
    from opi.fractionation_hydrogen import fractionation_hydrogen
    from opi.fractionation_oxygen import fractionation_oxygen
    
    # Test at various temperatures
    T_test = np.array([280.0, 273.15, 268.0, 258.0, 248.0, 238.0])
    
    alpha_h = fractionation_hydrogen(T_test)
    alpha_o = fractionation_oxygen(T_test)
    
    # Check values are reasonable
    # Hydrogen fractionation should be ~1.1 at 0°C
    assert alpha_h[1] > 1.1 and alpha_h[1] < 1.12, f"H fractionation at 0°C should be ~1.11, got {alpha_h[1]}"
    # Oxygen fractionation should be ~1.01 at 0°C
    assert alpha_o[1] > 1.01 and alpha_o[1] < 1.012, f"O fractionation at 0°C should be ~1.011, got {alpha_o[1]}"
    
    # Check that fractionation increases as temperature decreases
    assert alpha_h[-1] > alpha_h[0], "H fractionation should increase with decreasing temperature"
    assert alpha_o[-1] > alpha_o[0], "O fractionation should increase with decreasing temperature"
    
    print(f"  H fractionation at 0°C: {alpha_h[1]:.6f}")
    print(f"  O fractionation at 0°C: {alpha_o[1]:.6f}")
    print("  PASSED")


def test_base_state():
    """Test base state atmospheric calculation"""
    print("\nTesting base_state...")
    from opi.base_state import base_state
    
    # Test with typical parameters
    NM = 0.01  # rad/s
    T0 = 290.0  # K
    
    z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
    
    # Check outputs are reasonable
    assert len(z_bar) > 0, "z_bar should not be empty"
    assert T[0] == T0, "Surface temperature should match T0"
    assert rho_s0 > 0, "Surface vapor density should be positive"
    assert h_s > 0, "Scale height should be positive"
    assert gamma_ratio > 0, "Gamma ratio should be positive"
    
    print(f"  Surface vapor density: {rho_s0:.4f} kg/m^3")
    print(f"  Vapor scale height: {h_s:.0f} m")
    print(f"  Gamma ratio: {gamma_ratio:.3f}")
    print("  PASSED")


def test_wind_grid():
    """Test wind grid transformation"""
    print("\nTesting wind_grid...")
    from opi.fourier_solution import wind_grid
    
    # Create test grid
    x = np.linspace(-50000, 50000, 100)
    y = np.linspace(-50000, 50000, 100)
    azimuth = 90.0  # Eastward wind
    
    Sxy, Txy, s, t, Xst, Yst = wind_grid(x, y, azimuth)
    
    # Check outputs
    assert Sxy.shape == (len(y), len(x)), "Sxy shape should match input grid"
    assert len(s) > 0, "s vector should not be empty"
    assert len(t) > 0, "t vector should not be empty"
    
    print(f"  Grid size: {len(s)} x {len(t)}")
    print("  PASSED")


def test_fourier_solution():
    """Test Fourier solution for topography"""
    print("\nTesting fourier_solution...")
    from opi.fourier_solution import fourier_solution
    
    # Create simple test topography
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Parameters
    U = 10.0
    azimuth = 90.0
    NM = 0.01
    f_c = 1e-4
    h_rho = 8000.0
    
    result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
    
    # Check outputs
    assert 'h_wind' in result, "Result should contain h_wind"
    assert 'h_hat' in result, "Result should contain h_hat"
    assert 'k_z' in result, "Result should contain k_z"
    
    print(f"  h_wind shape: {result['h_wind'].shape}")
    print(f"  h_hat shape: {result['h_hat'].shape}")
    print("  PASSED")


def test_precipitation_grid():
    """Test precipitation grid calculation"""
    print("\nTesting precipitation_grid...")
    from opi.precipitation_grid import precipitation_grid
    from opi.base_state import base_state
    
    # Create test data
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    # Get base state
    z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(0.01, 290.0)
    
    # Run precipitation grid
    result = precipitation_grid(
        x, y, h_grid, U=10.0, azimuth=90.0, NM=0.01, f_c=1e-4,
        kappa=0.0, tau_c=1000.0, h_rho=h_rho, z_bar=z_bar, T=T,
        gamma_env=gamma_env, gamma_sat=gamma_sat, gamma_ratio=gamma_ratio,
        rho_s0=rho_s0, h_s=h_s, f_p0=1.0
    )
    
    # Check outputs
    assert 'p_grid' in result, "Result should contain p_grid"
    assert result['p_grid'].shape == h_grid.shape, "p_grid shape should match h_grid"
    assert result['tau_f'] > 0, "tau_f should be positive"
    
    print(f"  Precipitation range: {result['p_grid'].min():.6f} to {result['p_grid'].max():.6f} kg/m^2/s")
    print(f"  Mean fall time: {result['tau_f']:.1f} s")
    print("  PASSED")


def test_catchment_nodes():
    """Test catchment nodes calculation"""
    print("\nTesting catchment_nodes...")
    from opi.catchment_nodes import catchment_nodes
    
    # Create test grid
    x = np.linspace(0, 10000, 21)
    y = np.linspace(0, 10000, 21)
    X, Y = np.meshgrid(x, y)
    h_grid = 1000 * np.exp(-((X - 5000)**2 + (Y - 5000)**2) / (2 * 2000**2))
    
    # Test samples
    sample_x = np.array([5000, 3000, 7000])
    sample_y = np.array([5000, 3000, 7000])
    sample_lc = np.array(['L', 'C', 'C'])
    
    ij_catch, ptr_catch = catchment_nodes(sample_x, sample_y, sample_lc, x, y, h_grid)
    
    # Check outputs
    assert len(ij_catch) > 0, "ij_catch should not be empty"
    assert len(ptr_catch) == len(sample_x), "ptr_catch length should be n_samples"
    
    # Local sample should have 1 node
    local_nodes = ptr_catch[1] - ptr_catch[0]
    assert local_nodes == 1, f"Local sample should have 1 node, got {local_nodes}"
    
    print(f"  Total catchment nodes: {len(ij_catch)}")
    print(f"  Local sample nodes: {local_nodes}")
    print("  PASSED")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing OPI Core Physics Modules")
    print("=" * 60)
    
    tests = [
        test_saturated_vapor_pressure,
        test_fractionation,
        test_base_state,
        test_wind_grid,
        test_fourier_solution,
        test_precipitation_grid,
        test_catchment_nodes,
    ]
    
    failed = []
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "=" * 60)
    if not failed:
        print("All tests PASSED!")
    else:
        print(f"FAILED tests: {', '.join(failed)}")
    print("=" * 60)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
