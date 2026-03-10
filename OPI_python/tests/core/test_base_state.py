"""
Unit tests for the base_state module.

This module tests the atmospheric base state calculations, including
saturated vapor pressure and the vertical structure of the atmosphere.
"""

import numpy as np
import pytest

from opi.core.base_state import base_state, saturated_vapor_pressure


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def standard_temperature():
    """Return a standard sea-level temperature in Kelvin."""
    return 288.15  # 15°C in Kelvin


@pytest.fixture
def standard_buoyancy_frequency():
    """Return a standard saturated buoyancy frequency in rad/s."""
    return 0.01  # Typical value for moist atmosphere


@pytest.fixture
def base_state_result(standard_buoyancy_frequency, standard_temperature):
    """Return a base state result for testing."""
    return base_state(
        NM=standard_buoyancy_frequency,
        T0=standard_temperature,
        z_max=12e3,
        dz=100
    )


@pytest.fixture
def known_vapor_pressure_data():
    """Return known temperature-vapor pressure pairs for testing.
    
    Values calculated using the Buck (1981) formula.
    """
    return [
        (273.15, 611.2),      # 0°C
        (283.15, 1227.9),     # 10°C
        (293.15, 2338.8),     # 20°C
        (303.15, 4246.7),     # 30°C
        (313.15, 7377.3),     # 40°C
    ]


# =============================================================================
# Test Functions
# =============================================================================

def test_saturated_vapor_pressure(known_vapor_pressure_data):
    """Test vapor pressure calculation at known temperatures.
    
    Verifies that the saturated_vapor_pressure function returns values
    consistent with the Buck (1981) formula for the temperature range
    commonly encountered in atmospheric applications.
    
    Parameters
    ----------
    known_vapor_pressure_data : list
        List of (temperature_K, expected_pressure_Pa) tuples
    """
    for temp_k, expected_pa in known_vapor_pressure_data:
        result = saturated_vapor_pressure(temp_k)
        # Allow 1% tolerance for rounding differences
        assert result == pytest.approx(expected_pa, rel=0.01), (
            f"Vapor pressure at {temp_k}K: expected {expected_pa} Pa, "
            f"got {result} Pa"
        )


def test_saturated_vapor_pressure_array():
    """Test that vapor pressure calculation works with numpy arrays."""
    temps = np.array([273.15, 283.15, 293.15])
    results = saturated_vapor_pressure(temps)
    
    assert isinstance(results, np.ndarray)
    assert len(results) == len(temps)
    assert np.all(results > 0)


def test_base_state_output_structure(base_state_result):
    """Verify base_state returns dict with all expected keys.
    
    The base_state function should return a dictionary containing:
    - z_bar: elevation array
    - T: temperature profile
    - gamma_env: environmental lapse rate
    - gamma_sat: saturated adiabatic lapse rate
    - gamma_ratio: weighted mean of lapse rate ratio
    - rho_s0: surface water vapor density
    - h_s: water vapor scale height
    - rho0: surface total air density
    - h_rho: total air density scale height
    - p: pressure profile
    - r_s: saturation mixing ratio
    - rho_s: water vapor density profile
    - rho_total: total air density profile
    """
    expected_keys = {
        'z_bar', 'T', 'p', 'r_s',
        'gamma_env', 'gamma_sat', 'gamma_ratio',
        'rho_s0', 'h_s', 'rho0', 'h_rho',
        'rho_s', 'rho_total'
    }
    
    result_keys = set(base_state_result.keys())
    
    # Check all expected keys are present
    missing_keys = expected_keys - result_keys
    assert not missing_keys, f"Missing keys in base_state output: {missing_keys}"
    
    # Check no unexpected keys
    extra_keys = result_keys - expected_keys
    assert not extra_keys, f"Unexpected keys in base_state output: {extra_keys}"


def test_base_state_values(base_state_result):
    """Test that output values are physically reasonable.
    
    Verifies that the calculated atmospheric profiles fall within
    physically plausible ranges for Earth's atmosphere.
    """
    result = base_state_result
    
    # Temperature should decrease with height (typical range: 200-320 K)
    assert np.all(result['T'] > 150), "Temperature below 150 K (unrealistic)"
    assert np.all(result['T'] < 350), "Temperature above 350 K (unrealistic)"
    
    # Pressure should decrease from sea level (1000-103000 Pa range)
    assert result['p'][0] == pytest.approx(101325, rel=0.01), (
        "Surface pressure should be ~101325 Pa"
    )
    assert np.all(result['p'] > 1000), "Pressure below 1000 Pa (unrealistic)"
    assert np.all(result['p'] <= 101325), "Pressure exceeds surface pressure"
    
    # Mixing ratio should be positive and typically < 0.04 kg/kg
    assert np.all(result['r_s'] > 0), "Mixing ratio must be positive"
    assert np.all(result['r_s'] < 0.1), "Mixing ratio unrealistically high"
    
    # Water vapor density should be positive (typical range: 0.001-0.03 kg/m³)
    assert np.all(result['rho_s'] > 0), "Water vapor density must be positive"
    assert np.all(result['rho_s'] < 0.1), "Water vapor density unrealistically high"
    
    # Total air density should be positive (typical range: 0.3-1.3 kg/m³)
    assert np.all(result['rho_total'] > 0), "Total density must be positive"
    assert np.all(result['rho_total'] < 2.0), "Total density unrealistically high"
    
    # Surface water vapor density should be reasonable
    assert result['rho_s0'] > 0, "Surface vapor density must be positive"
    assert result['rho_s0'] < 0.05, "Surface vapor density unrealistically high"
    
    # Surface total density should be reasonable (~1.2 kg/m³ at sea level)
    assert result['rho0'] > 0.5, "Surface total density too low"
    assert result['rho0'] < 1.5, "Surface total density too high"


def test_base_state_stability_check():
    """Test that unstable atmospheres raise ValueError.
    
    An unstable atmosphere occurs when the environmental lapse rate
    becomes too small (gamma_env < 1e-3 K/m, equivalent to < 1°C/km).
    This happens when the saturated adiabatic lapse rate drops below
    this threshold, which occurs at very high temperatures (physically
    unrealistic for Earth's atmosphere but possible in extreme scenarios).
    
    At T0 = 400K (127°C), gamma_sat ≈ 0.0002 K/m, which triggers
    the stability check.
    """
    # Extreme temperature where gamma_sat drops below 0.001 K/m
    with pytest.raises(ValueError, match="gamma_env"):
        base_state(
            NM=0.001,  # Low buoyancy frequency (1 mrad/s)
            T0=400.0,  # Extreme surface temperature (127°C)
            z_max=6e3,  # Shorter domain
            dz=200
        )


def test_base_state_temperature_profile(base_state_result):
    """Verify temperature decreases with height.
    
    In a stable atmosphere, temperature should monotonically decrease
    with increasing altitude (environmental lapse rate is positive).
    """
    result = base_state_result
    z = result['z_bar']
    T = result['T']
    
    # Temperature should decrease with height
    temp_diff = np.diff(T)
    assert np.all(temp_diff <= 0), (
        "Temperature should decrease or remain constant with height"
    )
    
    # Verify the temperature profile length matches z_bar
    assert len(T) == len(z), (
        "Temperature array length should match elevation array length"
    )
    
    # Check that temperature at surface equals T0
    assert T[0] == pytest.approx(288.15, abs=0.01), (
        "Surface temperature should match input T0"
    )


def test_base_state_scale_heights(base_state_result):
    """Verify scale heights are positive and reasonable.
    
    Scale heights represent the e-folding distance for density decay.
    Typical values for Earth's atmosphere:
    - Water vapor scale height: ~2000-3000 m
    - Total density scale height: ~7000-8500 m
    """
    result = base_state_result
    
    # Both scale heights must be positive
    assert result['h_s'] > 0, "Water vapor scale height must be positive"
    assert result['h_rho'] > 0, "Total density scale height must be positive"
    
    # Water vapor scale height is typically 2-6 km
    assert result['h_s'] > 1000, "Water vapor scale height unrealistically low"
    assert result['h_s'] < 8000, "Water vapor scale height unrealistically high"
    
    # Total density scale height is typically 6-10 km
    assert result['h_rho'] > 5000, "Total density scale height unrealistically low"
    assert result['h_rho'] < 12000, "Total density scale height unrealistically high"
    
    # Water vapor scale height should be less than total density scale height
    # because water vapor is concentrated at lower altitudes
    assert result['h_s'] < result['h_rho'], (
        "Water vapor scale height should be less than total density scale height"
    )


def test_base_state_lapse_rates(base_state_result):
    """Test that lapse rates are physically consistent.
    
    The environmental lapse rate should be:
    1. Positive (temperature decreases with height)
    2. Less than the dry adiabatic lapse rate (~9.8 K/km)
    3. Close to but different from the saturated adiabatic lapse rate
    """
    result = base_state_result
    
    gamma_env = result['gamma_env']
    gamma_sat = result['gamma_sat']
    gamma_ratio = result['gamma_ratio']
    
    # Environmental lapse rate should be positive
    assert np.all(gamma_env > 0), "Environmental lapse rate must be positive"
    
    # Lapse rates should be less than dry adiabatic (~9.8 K/km = 0.0098 K/m)
    assert np.all(gamma_env < 0.02), "Environmental lapse rate too high"
    assert np.all(gamma_sat < 0.02), "Saturated lapse rate too high"
    
    # Environmental lapse rate should typically be less than saturated
    # (for stable moist atmosphere)
    assert np.all(gamma_env <= gamma_sat * 1.1), (
        "Environmental lapse rate should not exceed saturated by much"
    )
    
    # gamma_ratio should be positive and close to 1
    assert gamma_ratio > 0, "Gamma ratio must be positive"
    assert gamma_ratio < 2, "Gamma ratio unrealistically high"


def test_base_state_pressure_profile(base_state_result):
    """Test that pressure decreases exponentially with height."""
    result = base_state_result
    p = result['p']
    z = result['z_bar']
    
    # Pressure should decrease with height
    assert np.all(np.diff(p) < 0), "Pressure should decrease with height"
    
    # Pressure should be positive
    assert np.all(p > 0), "Pressure must be positive"
    
    # Verify scale height is consistent with pressure profile
    # ln(p/p0) ≈ -z/h, so h ≈ -z / ln(p/p0)
    if len(z) > 10:
        idx = len(z) // 2  # Use middle point
        estimated_scale_height = -z[idx] / np.log(p[idx] / p[0])
        assert estimated_scale_height > 5000, "Pressure scale height too low"
        assert estimated_scale_height < 12000, "Pressure scale height too high"


def test_base_state_grid_parameters():
    """Test that custom grid parameters are respected."""
    NM = 0.01
    T0 = 288.15
    z_max = 8000.0
    dz = 200.0
    
    result = base_state(NM=NM, T0=T0, z_max=z_max, dz=dz)
    
    # Check grid dimensions
    expected_n = int(np.round(z_max / dz)) + 1
    assert len(result['z_bar']) == expected_n, (
        f"Expected {expected_n} grid points, got {len(result['z_bar'])}"
    )
    
    # Check grid spacing
    assert result['z_bar'][1] - result['z_bar'][0] == dz, (
        "Grid spacing should match dz"
    )
    
    # Check maximum height
    assert result['z_bar'][-1] == pytest.approx(z_max, abs=dz), (
        f"Maximum height should be approximately {z_max}"
    )


def test_base_state_different_temperatures(standard_buoyancy_frequency):
    """Test base_state with different surface temperatures."""
    temperatures = [273.15, 288.15, 303.15]  # 0°C, 15°C, 30°C
    
    for T0 in temperatures:
        result = base_state(NM=standard_buoyancy_frequency, T0=T0, z_max=8e3, dz=200)
        
        # Surface temperature should match input
        assert result['T'][0] == pytest.approx(T0, abs=0.01), (
            f"Surface temperature should be {T0} K"
        )
        
        # Higher surface temperature should give higher water vapor density
        if T0 == temperatures[0]:
            rho_s0_ref = result['rho_s0']
        else:
            assert result['rho_s0'] > rho_s0_ref, (
                "Higher temperature should give higher surface vapor density"
            )


def test_base_state_different_buoyancy_frequencies(standard_temperature):
    """Test base_state with different buoyancy frequencies."""
    # Test with stable values that won't trigger instability error
    # Lower buoyancy frequencies and shorter domain to avoid instability
    buoyancy_freqs = [0.005, 0.008, 0.012]
    
    for NM in buoyancy_freqs:
        result = base_state(NM=NM, T0=standard_temperature, z_max=6e3, dz=200)
        
        # All basic sanity checks should pass
        assert np.all(result['T'] > 0)
        assert np.all(result['p'] > 0)
        assert result['gamma_ratio'] > 0


def test_base_state_exponential_fits(base_state_result):
    """Test that exponential fits for densities are accurate.
    
    The scale height parameters (rho_s0, h_s, rho0, h_rho) are obtained
    by fitting exponential profiles to the calculated densities.
    """
    result = base_state_result
    z = result['z_bar']
    
    # Test water vapor density exponential fit
    rho_s_fitted = result['rho_s0'] * np.exp(-z / result['h_s'])
    
    # Calculate relative error (use only points with significant vapor density)
    significant_mask = result['rho_s'] > 0.001 * result['rho_s0']
    if np.any(significant_mask):
        rel_error = np.abs(
            result['rho_s'][significant_mask] - rho_s_fitted[significant_mask]
        ) / result['rho_s'][significant_mask]
        mean_rel_error = np.mean(rel_error)
        
        # Exponential fit should be reasonably accurate
        assert mean_rel_error < 0.5, (
            f"Water vapor exponential fit has large mean relative error: {mean_rel_error}"
        )
    
    # Test total density exponential fit
    rho_total_fitted = result['rho0'] * np.exp(-z / result['h_rho'])
    
    # Calculate relative error
    rel_error_total = np.abs(result['rho_total'] - rho_total_fitted) / result['rho_total']
    mean_rel_error_total = np.mean(rel_error_total)
    
    # Exponential fit should be reasonably accurate
    assert mean_rel_error_total < 0.1, (
        f"Total density exponential fit has large mean relative error: {mean_rel_error_total}"
    )
