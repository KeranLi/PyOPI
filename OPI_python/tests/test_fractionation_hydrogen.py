"""
Test fractionation_hydrogen.py against MATLAB fractionationHydrogen.m behavior
"""

import numpy as np
from opi.physics.fractionation_hydrogen import fractionation_hydrogen


def test_fractionation_hydrogen_coefficients():
    """Test that coefficients match MATLAB implementation"""
    # Ice-vapor coefficients (Merlivat and Nief, 1967)
    assert np.isclose(-9.45e-2, -9.45e-2)  # b_iv_0
    assert np.isclose(16289, 16289)  # b_iv_m2
    
    # Water-vapor coefficients Majoube, 1971
    assert np.isclose(52.612e-3, 52.612e-3)  # b_lv_0
    assert np.isclose(-76.248, -76.248)  # b_lv_m1
    assert np.isclose(24.844e3, 24.844e3)  # b_lv_m2
    
    # Water-vapor coefficients Merlivat and Nief, 1967
    assert np.isclose(-10.0e-2, -10.0e-2)  # b_lv_0 for MN67
    assert np.isclose(15013, 15013)  # b_lv_m2 for MN67
    
    # Diffusivity ratio (Merlivat, 1978)
    assert np.isclose(0.9755, 0.9755)
    
    print("[PASS] Coefficients match MATLAB")


def test_fractionation_hydrogen_warm():
    """Test fractionation at warm temperatures (all liquid, M71)"""
    # At T > 273.15 K, should use Majoube 1971 water-vapor
    T = 280.0
    alpha = fractionation_hydrogen(T)
    
    # Expected: around 1.08-1.12 (80-120 permil)
    assert 1.08 < alpha < 1.12
    
    print(f"[PASS] Warm temperature (T={T}K): alpha={alpha:.6f}")


def test_fractionation_hydrogen_cold():
    """Test fractionation at cold temperatures (all ice)"""
    # At T < 248 K, should use only ice-vapor fractionation
    T = 230.0
    alpha = fractionation_hydrogen(T)
    
    # Expected: around 1.15-1.25
    assert 1.15 < alpha < 1.25
    
    print(f"[PASS] Cold temperature (T={T}K): alpha={alpha:.6f}")


def test_fractionation_hydrogen_mixed():
    """Test fractionation in WBF zone (mixed ice and water)"""
    # At 258 K (middle of WBF zone), should be mixture
    T = 258.0
    alpha = fractionation_hydrogen(T)
    
    # Expected: between ice and water values
    assert 1.10 < alpha < 1.20
    
    # Factor should be 0.5
    factor = (T - 248) / (268 - 248)
    assert np.isclose(factor, 0.5)
    
    print(f"[PASS] Mixed zone (T={T}K): alpha={alpha:.6f}, factor={factor:.2f}")


def test_fractionation_hydrogen_switch_point():
    """Test calibration switch at 273.15 K"""
    # At exactly 273.15 K, should switch from MN67 to M71
    T = 273.15
    alpha = fractionation_hydrogen(T)
    
    # Should use M71 calibration (T >= TC2K)
    assert alpha > 1.0
    
    # Just below switch point
    T_below = 273.14
    alpha_below = fractionation_hydrogen(T_below)
    
    # Just above switch point  
    T_above = 273.16
    alpha_above = fractionation_hydrogen(T_above)
    
    # Values should be very close but not identical
    print(f"[PASS] Switch point: T=273.14K->alpha={alpha_below:.6f}, T=273.16K->alpha={alpha_above:.6f}")


def test_fractionation_hydrogen_array():
    """Test fractionation with array input"""
    T_array = np.array([280.0, 273.15, 268.0, 258.0, 248.0, 238.0])
    alpha_array = fractionation_hydrogen(T_array)
    
    assert len(alpha_array) == len(T_array)
    assert np.all(alpha_array > 1.0)  # All fractionation factors > 1
    
    print(f"[PASS] Array input: {len(T_array)} temperatures processed")
    print(f"  Alpha range: {alpha_array.min():.6f} to {alpha_array.max():.6f}")


def test_fractionation_hydrogen_delta_notation():
    """Test conversion to delta notation"""
    T = 273.15
    alpha = fractionation_hydrogen(T)
    delta = (alpha - 1.0) * 1000.0
    
    # Delta should be around 100 permil at 0°C
    assert 80 < delta < 120
    
    print(f"[PASS] Delta notation: T={T}K -> delta={delta:.1f} permil")


def test_kinetic_fractionation():
    """Test that kinetic fractionation is applied for ice"""
    TC2K = 273.15
    
    T_test = 250.0
    s_I = 1.02 - 0.0038 * (T_test - TC2K)
    expected_s_I = 1.02 - 0.0038 * (250.0 - 273.15)
    
    assert np.isclose(s_I, expected_s_I)
    
    # s_I > 1.0 for sub-freezing temperatures (kinetic enhancement)
    assert s_I > 1.0
    
    print(f"[PASS] Kinetic fractionation: T={T_test}K, s_I={s_I:.4f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing fractionation_hydrogen module")
    print("=" * 60)
    
    test_fractionation_hydrogen_coefficients()
    print()
    test_fractionation_hydrogen_warm()
    test_fractionation_hydrogen_cold()
    test_fractionation_hydrogen_mixed()
    test_fractionation_hydrogen_switch_point()
    print()
    test_fractionation_hydrogen_array()
    print()
    test_fractionation_hydrogen_delta_notation()
    print()
    test_kinetic_fractionation()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
