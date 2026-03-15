"""
Test isotherm function against MATLAB isotherm.m behavior
"""

import numpy as np
from opi.physics.precipitation_grid import isotherm


def create_test_data(n_s=50, n_t=50, n_z=100):
    """Create test data for isotherm calculation"""
    # Grid dimensions
    n_s_grid = n_s
    n_t_grid = n_t
    
    # Base state elevation (increasing from 0 to 10000m)
    z_bar = np.linspace(0, 10000, n_z)
    
    # Temperature profile (decreasing with height, typical lapse rate)
    T = 288 - 0.0065 * z_bar  # From 288K at surface to ~223K at 10km
    
    # Lapse rates
    gamma_env = np.ones_like(z_bar) * 0.0065  # Environmental lapse rate
    gamma_sat = np.ones_like(z_bar) * 0.0050  # Saturated lapse rate
    
    # Density scale height
    h_rho = 8000.0
    
    # Fourier coefficients for simple topography (Gaussian hill)
    x = np.linspace(-50000, 50000, n_s_grid)
    y = np.linspace(-50000, 50000, n_t_grid)
    X, Y = np.meshgrid(x, y, indexing='ij')
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    h_hat = np.fft.fft2(h_grid)
    
    # Vertical wavenumber (simplified)
    k_s = 2 * np.pi * np.fft.fftfreq(n_s_grid, x[1] - x[0])
    k_t = 2 * np.pi * np.fft.fftfreq(n_t_grid, y[1] - y[0])
    K_S, K_T = np.meshgrid(k_s, k_t, indexing='ij')
    # Simplified k_z (would come from fourier_solution in real case)
    k_z = np.sqrt(np.maximum(0, K_S**2 + K_T**2))
    
    return {
        'z_bar': z_bar,
        'T': T,
        'gamma_env': gamma_env,
        'gamma_sat': gamma_sat,
        'h_rho': h_rho,
        'n_s': n_s_grid,
        'n_t': n_t_grid,
        'h_hat': h_hat,
        'k_z': k_z,
        'h_grid': h_grid
    }


def test_isotherm_basic():
    """Test basic isotherm calculation at 258K (freezing level)"""
    data = create_test_data()
    
    TR = 258.0  # Typical freezing level temperature
    
    z_R = isotherm(
        TR, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    print(f"Isotherm at {TR}K:")
    print(f"  Shape: {z_R.shape}")
    print(f"  Height range: {z_R.min():.1f} to {z_R.max():.1f} m")
    print(f"  Mean height: {z_R.mean():.1f} m")
    
    # Basic sanity checks
    assert z_R.shape == (data['n_s'], data['n_t'])
    assert not np.isnan(z_R).any()
    
    # For typical atmosphere, 258K is around 4.6km
    # With 2000m topography perturbation, should be in reasonable range
    expected_height = (288 - TR) / 0.0065  # ~4615m
    assert expected_height - 1000 < z_R.mean() < expected_height + 3000
    
    print("[PASS] Basic isotherm test")


def test_isotherm_223K():
    """Test isotherm at 223K (upper troposphere)"""
    data = create_test_data()
    
    TR = 223.0  # Upper troposphere
    
    z_R = isotherm(
        TR, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    print(f"Isotherm at {TR}K:")
    print(f"  Height range: {z_R.min():.1f} to {z_R.max():.1f} m")
    
    # 223K is higher than 258K
    z_258 = isotherm(
        258.0, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    assert z_R.mean() > z_258.mean(), "223K should be higher than 258K"
    
    print("[PASS] 223K isotherm test")


def test_isotherm_extrapolation():
    """Test behavior when TR is outside T range (extrapolation case)"""
    data = create_test_data()
    
    # T ranges from ~288K to ~223K
    # Test with TR = 300K (above surface, requires extrapolation)
    TR_high = 300.0
    z_R_high = isotherm(
        TR_high, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    # Test with TR = 200K (below top, requires extrapolation)
    TR_low = 200.0
    z_R_low = isotherm(
        TR_low, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    print(f"Extrapolation test:")
    print(f"  TR={TR_high}K: height range {z_R_high.min():.1f} to {z_R_high.max():.1f} m")
    print(f"  TR={TR_low}K: height range {z_R_low.min():.1f} to {z_R_low.max():.1f} m")
    
    # High TR should give negative or low height (below surface)
    # Low TR should give height above z_bar range
    assert z_R_high.mean() < z_R_low.mean()
    
    print("[PASS] Extrapolation test")


def test_isotherm_flat_terrain():
    """Test isotherm with flat terrain (h_hat ~ 0)"""
    data = create_test_data()
    
    # Zero topography
    data['h_hat'] = np.zeros_like(data['h_hat'])
    
    TR = 258.0
    z_R = isotherm(
        TR, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    print(f"Flat terrain isotherm at {TR}K:")
    print(f"  Height range: {z_R.min():.1f} to {z_R.max():.1f} m")
    print(f"  Std dev: {z_R.std():.4f} m")
    
    # Should be nearly constant (small numerical errors allowed)
    assert z_R.std() < 1.0, "Flat terrain should give nearly constant height"
    
    # Should match the expected base-state height
    expected_height = (288 - TR) / 0.0065
    assert abs(z_R.mean() - expected_height) < 100
    
    print("[PASS] Flat terrain test")


def test_isotherm_lapse_rate_ratio():
    """Test that lapse rate ratio affects the result correctly"""
    data = create_test_data()
    
    TR = 258.0
    
    # Standard case
    z_R_1 = isotherm(
        TR, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    # Case with gamma_sat = gamma_env (ratio = 1, perturbation should be zeroed)
    gamma_sat_equal = data['gamma_env'].copy()
    z_R_2 = isotherm(
        TR, data['z_bar'], data['T'], data['gamma_env'], gamma_sat_equal,
        data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
    )
    
    print(f"Lapse rate ratio test:")
    print(f"  gamma_sat/gamma_env = {data['gamma_sat'].mean()/data['gamma_env'].mean():.3f}: std = {z_R_1.std():.2f}")
    print(f"  gamma_sat/gamma_env = 1.000: std = {z_R_2.std():.4f}")
    
    # When gamma_sat = gamma_env, (1 - ratio) = 0, so z_R should be flat
    # But wait, z_bar_R is constant, so z_R should be constant
    assert z_R_2.std() < 1.0, "When gamma_sat=gamma_env, result should be nearly flat"
    
    print("[PASS] Lapse rate ratio test")


def test_isotherm_temperature_sensitivity():
    """Test that isotherm height decreases with increasing temperature"""
    data = create_test_data()
    
    temperatures = [243, 253, 263, 273, 283]
    heights = []
    
    for TR in temperatures:
        z_R = isotherm(
            TR, data['z_bar'], data['T'], data['gamma_env'], data['gamma_sat'],
            data['h_rho'], data['n_s'], data['n_t'], data['h_hat'], data['k_z']
        )
        heights.append(z_R.mean())
        print(f"  T={TR}K: mean height={z_R.mean():.1f}m")
    
    # Higher temperature should give lower height
    for i in range(len(heights) - 1):
        assert heights[i] > heights[i+1], "Higher T should give lower height"
    
    print("[PASS] Temperature sensitivity test")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing isotherm function")
    print("=" * 60)
    
    test_isotherm_basic()
    print()
    test_isotherm_223K()
    print()
    test_isotherm_extrapolation()
    print()
    test_isotherm_flat_terrain()
    print()
    test_isotherm_lapse_rate_ratio()
    print()
    test_isotherm_temperature_sensitivity()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
