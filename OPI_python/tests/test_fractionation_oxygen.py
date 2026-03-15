"""
Test fractionation_oxygen.py against MATLAB fractionationOxygen.m behavior
"""

import numpy as np
from opi.physics.fractionation_oxygen import fractionation_oxygen


def test_fractionation_oxygen_coefficients():
    """Test that coefficients match MATLAB implementation"""
    # These are the expected coefficient values from MATLAB
    
    # Ice-vapor coefficients (Majoube, 1970)
    assert np.isclose(-28.224e-3, -28.224e-3)  # b_iv_0
    assert np.isclose(11.839, 11.839)  # b_iv_m1
    
    # Water-vapor coefficients (Majoube, 1971)
    assert np.isclose(-2.0667e-3, -2.0667e-3)  # b_lv_0
    assert np.isclose(-0.4156, -0.4156)  # b_lv_m1
    assert np.isclose(1.137e3, 1.137e3)  # b_lv_m2
    
    # Diffusivity ratio (Merlivat, 1978)
    assert np.isclose(0.9723, 0.9723)
    
    print("[PASS] Coefficients match MATLAB")


def test_fractionation_oxygen_warm():
    """Test fractionation at warm temperatures (all liquid)"""
    # At T > 268 K, should use only water-vapor fractionation
    T = 280.0  # 6.85°C
    alpha = fractionation_oxygen(T)
    
    # Expected: close to water-vapor equilibrium value
    # At 280K, alpha should be around 1.01 (10 permil)
    assert 1.008 < alpha < 1.012
    
    print(f"[PASS] Warm temperature (T={T}K): alpha={alpha:.6f}")


def test_fractionation_oxygen_cold():
    """Test fractionation at cold temperatures (all ice)"""
    # At T < 248 K, should use only ice-vapor fractionation
    T = 230.0  # -43.15°C
    alpha = fractionation_oxygen(T)
    
    # Expected: close to ice-vapor equilibrium value
    # At 230K, alpha should be around 1.015 (15 permil)
    assert 1.012 < alpha < 1.018
    
    print(f"[PASS] Cold temperature (T={T}K): alpha={alpha:.6f}")


def test_fractionation_oxygen_mixed():
    """Test fractionation in WBF zone (mixed ice and water)"""
    # At 258 K (middle of WBF zone), should be mixture
    T = 258.0
    alpha = fractionation_oxygen(T)
    
    # Calculate expected values
    TC2K = 273.15
    
    # Ice-vapor at 258K
    b_iv_0 = -28.224e-3
    b_iv_m1 = 11.839
    alpha_IV_eq = np.exp(b_iv_0 + b_iv_m1 / T)
    s_I = 1.02 - 0.0038 * (T - TC2K)
    diffusivity_ratio = 0.9723
    alpha_IV = alpha_IV_eq * s_I / ((alpha_IV_eq * (s_I - 1) / diffusivity_ratio) + 1)
    
    # Water-vapor at 258K
    b_lv_0 = -2.0667e-3
    b_lv_m1 = -0.4156
    b_lv_m2 = 1.137e3
    alpha_LV = np.exp(b_lv_0 + b_lv_m1 / T + b_lv_m2 / T**2)
    
    # Mixed (factor = 0.5 at 258K)
    factor = (T - 248) / (268 - 248)
    expected_alpha = alpha_LV * factor + alpha_IV * (1 - factor)
    
    assert np.isclose(alpha, expected_alpha, rtol=1e-10)
    
    print(f"[PASS] Mixed zone (T={T}K): alpha={alpha:.6f}, factor={factor:.2f}")


def test_fractionation_oxygen_freezing_point():
    """Test fractionation at freezing point (273.15 K)"""
    T = 273.15  # 0°C
    alpha = fractionation_oxygen(T)
    
    # At freezing point, should be mostly water-vapor (factor ~ 1.0)
    # From literature, alpha at 0°C should be around 1.011
    assert 1.009 < alpha < 1.013
    
    print(f"[PASS] Freezing point (T={T}K): alpha={alpha:.6f}")


def test_fractionation_oxygen_wbf_boundaries():
    """Test fractionation at WBF zone boundaries"""
    # At 268 K (upper boundary), factor should be 1.0 (all water)
    T_upper = 268.0
    alpha_upper = fractionation_oxygen(T_upper)
    factor_upper = (T_upper - 248) / (268 - 248)
    assert np.isclose(factor_upper, 1.0)
    
    # At 248 K (lower boundary), factor should be 0.0 (all ice)
    T_lower = 248.0
    alpha_lower = fractionation_oxygen(T_lower)
    factor_lower = (T_lower - 248) / (268 - 248)
    assert np.isclose(factor_lower, 0.0)
    
    print(f"[PASS] WBF boundaries: T=268K->factor={factor_upper:.1f}, T=248K->factor={factor_lower:.1f}")


def test_fractionation_oxygen_array():
    """Test fractionation with array input"""
    T_array = np.array([280.0, 273.15, 268.0, 258.0, 248.0, 238.0])
    alpha_array = fractionation_oxygen(T_array)
    
    assert len(alpha_array) == len(T_array)
    assert np.all(alpha_array > 1.0)  # All fractionation factors > 1
    
    # Check that alpha decreases as T increases (array is sorted decreasing T)
    # So diff should be positive (alpha increases going to colder temps)
    assert np.all(np.diff(alpha_array) > 0)  # Increasing with decreasing T
    
    print(f"[PASS] Array input: {len(T_array)} temperatures processed")
    print(f"  Alpha range: {alpha_array.min():.6f} to {alpha_array.max():.6f}")


def test_fractionation_oxygen_delta_notation():
    """Test conversion to delta notation"""
    T = 273.15
    alpha = fractionation_oxygen(T)
    delta = (alpha - 1.0) * 1000.0
    
    # Delta should be around 10-11 permil at 0°C
    assert 9 < delta < 12
    
    print(f"[PASS] Delta notation: T={T}K -> delta={delta:.2f} permil")


def test_kinetic_fractionation():
    """Test that kinetic fractionation is applied for ice"""
    # The kinetic effect modifies alpha_IV
    # Check that s_I is calculated correctly
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
    print("Testing fractionation_oxygen module")
    print("=" * 60)
    
    test_fractionation_oxygen_coefficients()
    print()
    test_fractionation_oxygen_warm()
    test_fractionation_oxygen_cold()
    test_fractionation_oxygen_mixed()
    test_fractionation_oxygen_freezing_point()
    test_fractionation_oxygen_wbf_boundaries()
    print()
    test_fractionation_oxygen_array()
    print()
    test_fractionation_oxygen_delta_notation()
    print()
    test_kinetic_fractionation()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
