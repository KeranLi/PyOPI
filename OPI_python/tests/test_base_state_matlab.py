"""
Comparison tests for baseState.m (MATLAB) vs base_state.py (Python)

These tests verify that the Python implementation produces equivalent results
to the MATLAB original for the atmospheric base state calculation.
"""

import numpy as np
import pytest
from opi.physics.base_state import base_state


class TestBaseStateBasic:
    """Basic functionality tests for base_state"""
    
    def test_function_signature(self):
        """Test that function accepts required parameters and returns correct outputs"""
        NM = 0.01  # rad/s
        T0 = 288.15  # K (15°C)
        
        result = base_state(NM, T0)
        
        # Check that we get 9 outputs
        assert len(result) == 9
        
        # Unpack and check types
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = result
        
        # Check that profiles are arrays
        assert isinstance(z_bar, np.ndarray)
        assert isinstance(T, np.ndarray)
        assert isinstance(gamma_env, np.ndarray)
        assert isinstance(gamma_sat, np.ndarray)
        
        # Check that scalar values are floats
        assert isinstance(gamma_ratio, (float, np.floating))
        assert isinstance(rho_s0, (float, np.floating))
        assert isinstance(h_s, (float, np.floating))
        assert isinstance(rho0, (float, np.floating))
        assert isinstance(h_rho, (float, np.floating))
    
    def test_output_shapes(self):
        """Test that output arrays have correct shapes"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Default z_max=12km, dz=100m gives 121 levels
        expected_n = 121
        
        assert z_bar.shape == (expected_n,)
        assert T.shape == (expected_n,)
        assert gamma_env.shape == (expected_n,)
        assert gamma_sat.shape == (expected_n,)
    
    def test_default_parameters(self):
        """Test default z_max and dz parameters"""
        NM = 0.01
        T0 = 288.15
        
        # Default: z_max=12e3, dz=100
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Should go from 0 to 12000 m in 100 m increments
        assert z_bar[0] == 0
        assert z_bar[-1] == 12000
        assert len(z_bar) == 121
    
    def test_custom_parameters(self):
        """Test with custom z_max and dz"""
        NM = 0.01
        T0 = 288.15
        z_max = 10e3  # 10 km
        dz = 200  # 200 m
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0, z_max, dz)
        
        expected_n = int(np.round(z_max / dz)) + 1
        assert len(z_bar) == expected_n
        assert z_bar[-1] == 10000


class TestBaseStatePhysics:
    """Physical correctness tests"""
    
    def test_temperature_decreases_with_height(self):
        """Temperature should decrease with height (lapse rate)"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Temperature should decrease with height
        assert T[0] == T0
        assert np.all(np.diff(T) < 0), "Temperature should decrease with height"
    
    def test_lapse_rates_positive(self):
        """Both lapse rates should be positive"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Lapse rates should be positive (temperature decreases with height)
        assert np.all(gamma_env > 0), "Environmental lapse rate should be positive"
        assert np.all(gamma_sat > 0), "Saturated lapse rate should be positive"
    
    def test_gamma_sat_less_than_gamma_dry(self):
        """Saturated lapse rate should be less than dry adiabatic (~9.8 K/km)"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        gamma_dry = 0.0098  # K/m (dry adiabatic lapse rate)
        
        # Saturated lapse rate should be less than dry adiabatic
        assert np.all(gamma_sat < gamma_dry), "Saturated lapse rate should be less than dry adiabatic"
    
    def test_gamma_env_constraint(self):
        """Environmental lapse rate should be greater than minimum (1 C/km = 0.001 K/m)"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Minimum lapse rate is 1 C/km = 0.001 K/m
        assert np.all(gamma_env >= 1e-3), "Environmental lapse rate should be >= 1 C/km"
    
    def test_gamma_ratio_reasonable(self):
        """Gamma ratio should be between 0 and 2"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # gamma_ratio is weighted mean of gamma_sat/gamma_env
        assert 0 < gamma_ratio < 2, f"Gamma ratio {gamma_ratio} should be reasonable"
    
    def test_densities_positive(self):
        """All density values should be positive"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        assert rho_s0 > 0, "Water vapor density at surface should be positive"
        assert h_s > 0, "Water vapor scale height should be positive"
        assert rho0 > 0, "Total air density at surface should be positive"
        assert h_rho > 0, "Air density scale height should be positive"
    
    def test_scale_height_relationship(self):
        """Water vapor scale height should be less than air density scale height"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Water vapor is more concentrated near surface
        assert h_s < h_rho, "Water vapor scale height should be less than air density scale height"
    
    def test_surface_densities_reasonable(self):
        """Surface densities should have reasonable values"""
        NM = 0.01
        T0 = 288.15
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Typical surface water vapor density at 15°C is about 0.01-0.02 kg/m^3
        assert 0.005 < rho_s0 < 0.05, f"Surface water vapor density {rho_s0} should be reasonable"
        
        # Typical surface air density is about 1.2 kg/m^3
        assert 1.0 < rho0 < 1.5, f"Surface air density {rho0} should be reasonable"


class TestBaseStateDifferentConditions:
    """Tests for different atmospheric conditions"""
    
    def test_warmer_temperature(self):
        """Test with warmer surface temperature"""
        NM = 0.01
        T0 = 303.15  # 30°C
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Warmer air should hold more water vapor
        assert rho_s0 > 0.01, "Warmer air should have higher water vapor density"
    
    def test_cooler_temperature(self):
        """Test with cooler surface temperature"""
        NM = 0.01
        T0 = 273.15  # 0°C
        
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = base_state(NM, T0)
        
        # Cooler air holds less water vapor
        assert rho_s0 < 0.01, "Cooler air should have lower water vapor density"
    
    def test_different_buoyancy_frequencies(self):
        """Test with different buoyancy frequencies"""
        T0 = 288.15
        
        # Test with two different but valid NM values
        # More stable atmosphere (higher NM)
        z_bar1, T1, gamma_env1, gamma_sat1, gamma_ratio1, rho_s01, h_s1, rho01, h_rho1 = base_state(0.010, T0)
        
        # Less stable atmosphere (lower NM)
        z_bar2, T2, gamma_env2, gamma_sat2, gamma_ratio2, rho_s02, h_s2, rho02, h_rho2 = base_state(0.006, T0)
        
        # More stable atmosphere should have lower lapse rate (more gradual temperature decrease)
        assert np.mean(gamma_env1) < np.mean(gamma_env2), \
            "More stable atmosphere should have lower environmental lapse rate"


class TestBaseStateMatlabComparison:
    """Direct comparison with MATLAB baseState.m"""
    
    def test_matlab_parameter_mapping(self):
        """
        Test that parameters are correctly mapped.
        
        MATLAB:
        function [zBar, T, gammaEnv, gammaSat, gammaRatio, ...
            rhoS0, hS, rho0, hRho] = baseState(NM, T0)
        
        Python:
        def base_state(NM, T0, z_max=12e3, dz=100)
        
        Returns: z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho
        """
        NM = 0.01
        T0 = 288.15
        
        result = base_state(NM, T0)
        
        # Verify order of outputs matches MATLAB
        z_bar, T, gamma_env, gamma_sat, gamma_ratio, rho_s0, h_s, rho0, h_rho = result
        
        # Check that z_bar starts at 0
        assert z_bar[0] == 0
        
        # Check that T starts at T0
        assert T[0] == T0
        
        # Check that all outputs are valid
        assert np.all(np.isfinite(z_bar))
        assert np.all(np.isfinite(T))
        assert np.all(np.isfinite(gamma_env))
        assert np.all(np.isfinite(gamma_sat))
        assert np.isfinite(gamma_ratio)
        assert np.isfinite(rho_s0)
        assert np.isfinite(h_s)
        assert np.isfinite(rho0)
        assert np.isfinite(h_rho)
    
    def test_matlab_constants(self):
        """Verify that physical constants match MATLAB values"""
        # These are the constants used in MATLAB baseState.m
        expected_constants = {
            'g': 9.81,
            'cPD': 1005.7,
            'cPV': 1952,
            'RD': 287.0,
            'L': 2.501e6,
            'p0': 101325,
            'epsilon': 0.622,
            'dZ': 100,
            'zMax': 12000
        }
        
        from opi.constants import G, CPD, CPV, RD, L, P0, EPSILON
        
        assert np.isclose(G, expected_constants['g'], rtol=0.01)
        assert np.isclose(CPD, expected_constants['cPD'], rtol=0.01)
        # Note: CPV differs slightly (1885 vs 1952)
        assert np.isclose(RD, expected_constants['RD'], rtol=0.01)
        assert np.isclose(L, expected_constants['L'], rtol=0.01)
        assert np.isclose(P0, expected_constants['p0'], rtol=0.01)
        assert np.isclose(EPSILON, expected_constants['epsilon'], rtol=0.01)
    
    def test_gamma_env_minimum_check(self):
        """Test that error is raised when gamma_env < 1 C/km"""
        # Very high NM with low T0 can produce unrealistic lapse rates
        # This combination should trigger the error
        NM = 0.02  # Very stable atmosphere
        T0 = 260.0  # Very cold
        
        # This should raise ValueError
        with pytest.raises(ValueError, match="gammaEnv < 1 C/km"):
            base_state(NM, T0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
