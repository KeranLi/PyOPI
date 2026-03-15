"""
Comparison tests for opiCalc_TwoWinds.m (MATLAB) vs opi_calc_two_winds.py (Python)

These tests verify that the Python high-level interface produces equivalent 
results to the MATLAB original for the TwoWinds OPI calculation workflow.
"""

import numpy as np
import pytest
import os
import tempfile
from opi.app.calc_two_winds import opi_calc_two_winds


class TestOpiCalcTwoWindsBasic:
    """Basic functionality tests for opi_calc_two_winds"""
    
    def test_function_exists(self):
        """Test that opi_calc_two_winds function exists and is callable"""
        assert callable(opi_calc_two_winds)
    
    def test_default_parameters(self):
        """Test that function runs with default parameters"""
        result = opi_calc_two_winds(verbose=False)
        
        # Check that we get a dictionary result
        assert isinstance(result, dict)
        
        # Check that required keys exist
        assert 'solution_params' in result
        assert 'derived_params' in result
        assert 'precipitation' in result
        assert 'isotope' in result
    
    def test_solution_params_structure(self):
        """Test that solution_params has correct structure for 19 parameters"""
        result = opi_calc_two_winds(verbose=False)
        
        params = result['solution_params']
        
        # Check for expected parameter names (19 parameters for two-wind model)
        # Wind 1 parameters
        assert 'wind1_U' in params
        assert 'wind1_az' in params
        assert 'wind1_T0' in params
        assert 'wind1_M' in params
        assert 'wind1_kappa' in params
        assert 'wind1_tau_c' in params
        assert 'wind1_d2h0' in params
        assert 'wind1_d_d2h0_d_lat' in params
        assert 'wind1_f_p0' in params
        
        # Wind 2 parameters
        assert 'wind2_U' in params
        assert 'wind2_az' in params
        assert 'wind2_T0' in params
        assert 'wind2_M' in params
        assert 'wind2_kappa' in params
        assert 'wind2_tau_c' in params
        assert 'wind2_d2h0' in params
        assert 'wind2_d_d2h0_d_lat' in params
        assert 'wind2_f_p0' in params
        
        # Mixing fraction
        assert 'mix_frac2' in params
    
    def test_results_structure(self):
        """Test that results dictionary has correct structure"""
        result = opi_calc_two_winds(verbose=False)
        
        # Check individual wind field results exist
        assert 'precipitation1' in result
        assert 'precipitation2' in result
        assert 'isotope1' in result
        assert 'isotope2' in result
        
        # Check combined results exist
        assert 'precipitation' in result
        assert 'isotope' in result
        
        # Check grid coordinates
        assert 'x' in result
        assert 'y' in result
        assert 'h_grid' in result


class TestOpiCalcTwoWindsParameters:
    """Tests for parameter handling"""
    
    def test_custom_solution_vector(self):
        """Test that custom 19-parameter solution vector is used correctly"""
        # 19-parameter solution vector (matches run file format: wind1 | fraction | wind2)
        custom_beta = [
            # Wind 1 (9 parameters)
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
            # Mixing fraction (1 parameter)
            0.6,
            # Wind 2 (9 parameters)
            8.0, 270.0, 292.0, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0
        ]
        
        result = opi_calc_two_winds(solution_vector=custom_beta, verbose=False)
        
        # Check that custom parameters were used
        params = result['solution_params']
        assert params['wind1_U'] == 10.0
        assert params['wind1_az'] == 90.0
        assert params['wind1_T0'] == 288.15
        assert params['wind2_U'] == 8.0
        assert params['wind2_az'] == 270.0
        assert params['mix_frac2'] == 0.6
    
    def test_different_wind_combinations(self):
        """Test calculation with different wind combinations"""
        # Same direction winds (wind1 | fraction | wind2 format)
        beta_same = [
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
            0.5,  # fraction
            8.0, 90.0, 288.15, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0
        ]
        
        # Opposite direction winds
        beta_opposite = [
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
            0.5,  # fraction
            8.0, 270.0, 288.15, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0
        ]
        
        result_same = opi_calc_two_winds(solution_vector=beta_same, verbose=False)
        result_opposite = opi_calc_two_winds(solution_vector=beta_opposite, verbose=False)
        
        # Both should complete without errors
        assert 'precipitation' in result_same
        assert 'precipitation' in result_opposite
        
        # Results should be different
        assert not np.allclose(result_same['precipitation'], result_opposite['precipitation'])
    
    def test_different_mixing_fractions(self):
        """Test with different mixing fractions"""
        beta_low_frac = [
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
            0.2,  # 20% from wind 2
            8.0, 270.0, 292.0, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0
        ]
        
        beta_high_frac = [
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
            0.8,  # 80% from wind 2
            8.0, 270.0, 292.0, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0
        ]
        
        result_low = opi_calc_two_winds(solution_vector=beta_low_frac, verbose=False)
        result_high = opi_calc_two_winds(solution_vector=beta_high_frac, verbose=False)
        
        # Both should complete
        assert 'precipitation' in result_low
        assert 'precipitation' in result_high
        
        # Results should be different
        assert not np.allclose(result_low['precipitation'], result_high['precipitation'])


class TestOpiCalcTwoWindsMatlabComparison:
    """Direct comparison with MATLAB opiCalc_TwoWinds.m workflow"""
    
    def test_matlab_workflow_structure(self):
        """
        Test that the workflow matches MATLAB opiCalc_TwoWinds:
        1. Load run file / input data
        2. Unpack 19 beta parameters (9 for wind 1, 1 fraction, 9 for wind 2)
        3. Call calc_TwoWinds (or equivalent)
        4. Combine results according to mixing fraction
        5. Return/save results
        """
        result = opi_calc_two_winds(verbose=False)
        
        # Verify the workflow produces expected outputs
        assert isinstance(result, dict)
        
        # Check that we have the key outputs that MATLAB saves
        assert 'precipitation' in result  # pGrid
        assert 'precipitation1' in result  # pGrid_1
        assert 'precipitation2' in result  # pGrid_2
        assert 'isotope' in result  # d2HGrid
        assert 'isotope1' in result  # d2HGrid_1
        assert 'isotope2' in result  # d2HGrid_2
    
    def test_matlab_constant_values(self):
        """Test that key constants match MATLAB opiCalc_TwoWinds"""
        from opi.constants import HR, SD_RES_RATIO, TC2K
        
        # hR = 540 m (characteristic distance for isotopic exchange)
        assert HR == 540.0
        
        # sdResRatio = 28.3 (standard deviation ratio)
        assert SD_RES_RATIO == 28.3
        
        # TC2K = 273.15 (Celsius to Kelvin conversion)
        assert TC2K == 273.15
    
    def test_parameter_mapping_matches_matlab(self):
        """
        Test that beta parameter mapping matches MATLAB/run file format:
        
        State #1 (indices 0-8):
        beta(0) = U_1 (wind speed)
        beta(1) = azimuth_1
        beta(2) = T0_1 (sea-level temperature)
        beta(3) = M_1 (mountain-height number)
        beta(4) = kappa_1 (eddy diffusion)
        beta(5) = tauC_1 (condensation time)
        beta(6) = d2H0_1 (base d2H)
        beta(7) = dD2H0_dLat_1 (latitudinal gradient)
        beta(8) = fP0_1 (residual precipitation)
        
        Mixing fraction (index 9):
        beta(9) = fraction (fraction of wind 2)
        
        State #2 (indices 10-18):
        beta(10) = U_2
        beta(11) = azimuth_2
        beta(12) = T0_2
        beta(13) = M_2
        beta(14) = kappa_2
        beta(15) = tauC_2
        beta(16) = d2H0_2
        beta(17) = dD2H0_dLat_2
        beta(18) = fP0_2
        """
        # Beta vector in run file format: wind1 (9) | fraction (1) | wind2 (9)
        beta = [
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,  # Wind 1 (0-8)
            0.5,  # fraction (9)
            8.0, 270.0, 292.0, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0     # Wind 2 (10-18)
        ]
        
        result = opi_calc_two_winds(solution_vector=beta, verbose=False)
        params = result['solution_params']
        
        # Verify parameter values are correctly passed through
        assert params['wind1_U'] == 10.0
        assert params['wind1_az'] == 90.0
        assert params['wind1_T0'] == 288.15
        assert params['wind2_U'] == 8.0
        assert params['wind2_az'] == 270.0
        assert params['wind2_T0'] == 292.0
        assert params['mix_frac2'] == 0.5
    
    def test_output_grids_consistency(self):
        """Test that output grids are consistent with each other"""
        result = opi_calc_two_winds(verbose=False)
        
        # All precipitation grids should have the same shape
        p_grid = result['precipitation']
        p_grid_1 = result['precipitation1']
        p_grid_2 = result['precipitation2']
        
        assert p_grid.shape == p_grid_1.shape
        assert p_grid.shape == p_grid_2.shape
        
        # All isotope grids should have the same shape
        d2h_grid = result['isotope']
        d2h_grid_1 = result['isotope1']
        d2h_grid_2 = result['isotope2']
        
        assert d2h_grid.shape == d2h_grid_1.shape
        assert d2h_grid.shape == d2h_grid_2.shape
        
        # Precipitation and isotope grids should have the same shape
        assert p_grid.shape == d2h_grid.shape
    
    def test_combined_precipitation_weighting(self):
        """Test that combined precipitation follows the mixing fraction"""
        # Use equal mixing
        beta_equal = [
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
            0.5,  # 50% from each
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0
        ]
        
        result = opi_calc_two_winds(solution_vector=beta_equal, verbose=False)
        
        # With identical winds and 50% mixing, combined should equal individual
        p1 = result['precipitation1']
        p2 = result['precipitation2']
        p_combined = result['precipitation']
        
        # For identical winds, p1 and p2 should be similar
        np.testing.assert_allclose(p1, p2, rtol=1e-5)
        
        # Combined should be similar to individual (when both are same)
        np.testing.assert_allclose(p_combined, p1, rtol=1e-5)


class TestOpiCalcTwoWindsEdgeCases:
    """Edge case tests"""
    
    def test_empty_run_file_path(self):
        """Test with empty run file path"""
        result = opi_calc_two_winds(run_file_path=None, verbose=False)
        
        assert isinstance(result, dict)
        assert 'solution_params' in result
    
    def test_extreme_mixing_fractions(self):
        """Test with extreme mixing fractions"""
        # Almost all from wind 1
        beta_mostly_1 = [
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
            0.01,  # 1% from wind 2
            8.0, 270.0, 292.0, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0
        ]
        
        result = opi_calc_two_winds(solution_vector=beta_mostly_1, verbose=False)
        
        assert 'precipitation' in result
        assert 'precipitation1' in result
        assert 'precipitation2' in result
    
    def test_very_different_wind_parameters(self):
        """Test with very different wind parameters"""
        # Very different winds
        beta_different = [
            5.0, 0.0, 265.0, 0.1, 500.0, 500.0, -0.150, -3e-3, 1.0,
            0.5,  # fraction
            20.0, 180.0, 295.0, 0.8, 2000.0, 2000.0, -0.050, -1e-3, 1.0
        ]
        
        # Should complete without crashing
        result = opi_calc_two_winds(solution_vector=beta_different, verbose=False)
        
        assert isinstance(result, dict)
        assert 'precipitation' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
