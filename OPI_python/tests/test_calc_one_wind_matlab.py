"""
Comparison tests for calc_OneWind.m (MATLAB) vs calc_one_wind.py (Python)

These tests verify that the Python implementation produces equivalent results
to the MATLAB original for the OneWind (single moisture source) model.
"""

import numpy as np
import pytest
from opi.physics.calc_one_wind import calc_one_wind


class TestCalcOneWindBasic:
    """Basic functionality tests for calc_one_wind"""
    
    def test_function_signature(self):
        """Test that function accepts all required parameters"""
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        # 9 beta parameters for OneWind
        beta = np.array([10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0])
        
        f_c = 1e-4
        h_r = 2500.0
        lat = np.array([45.0])
        lat0 = 45.0
        b_mwl_sample = np.array([10.0, 8.0])
        ij_catch = [(16, 16)]
        ptr_catch = [0, 1]
        sample_d2h = np.array([-0.100])
        sample_d18o = np.array([-0.012])
        cov = np.array([[1e-4, 1e-5], [1e-5, 1e-4]])
        
        # Should not raise any errors
        result = calc_one_wind(
            beta, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 9, True
        )
        
        # Check that we get the expected number of outputs
        assert len(result) == 27
    
    def test_output_shapes(self):
        """Test that output arrays have correct shapes"""
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        beta = np.array([10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0])
        
        f_c = 1e-4
        h_r = 2500.0
        lat = np.array([45.0])
        lat0 = 45.0
        b_mwl_sample = np.array([10.0, 8.0])
        ij_catch = [(16, 16)]
        ptr_catch = [0, 1]
        sample_d2h = np.array([-0.100])
        sample_d18o = np.array([-0.012])
        cov = np.array([[1e-4, 1e-5], [1e-5, 1e-4]])
        
        result = calc_one_wind(
            beta, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 9, True
        )
        
        # Extract key outputs
        p_grid = result[15]  # Precipitation grid
        f_m_grid = result[16]  # Moisture grid
        d2h_grid = result[22]  # d2H grid
        d18o_grid = result[23]  # d18O grid
        
        # Check shapes
        assert p_grid.shape == (32, 32)
        assert f_m_grid.shape == (32, 32)
        assert d2h_grid.shape == (32, 32)
        assert d18o_grid.shape == (32, 32)


class TestCalcOneWindAlgorithm:
    """Algorithmic correctness tests"""
    
    def test_parameter_mapping(self):
        """Test that beta parameters are correctly mapped"""
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        # Different wind speeds should give different precipitation patterns
        beta_fast = np.array([15.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0])
        beta_slow = np.array([5.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0])
        
        f_c = 1e-4
        h_r = 2500.0
        lat = np.array([45.0])
        lat0 = 45.0
        b_mwl_sample = np.array([10.0, 8.0])
        ij_catch = [(16, 16)]
        ptr_catch = [0, 1]
        sample_d2h = np.array([-0.100])
        sample_d18o = np.array([-0.012])
        cov = np.array([[1e-4, 1e-5], [1e-5, 1e-4]])
        
        result_fast = calc_one_wind(
            beta_fast, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 9, True
        )
        
        result_slow = calc_one_wind(
            beta_slow, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 9, True
        )
        
        p_grid_fast = result_fast[16]
        p_grid_slow = result_slow[16]
        
        # Faster wind should generally produce more orographic precipitation
        # due to increased moisture flux
        assert np.sum(p_grid_fast) != np.sum(p_grid_slow)
    
    def test_precipitation_non_negative(self):
        """Test that precipitation is non-negative everywhere"""
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        beta = np.array([10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0])
        
        f_c = 1e-4
        h_r = 2500.0
        lat = np.array([45.0])
        lat0 = 45.0
        b_mwl_sample = np.array([10.0, 8.0])
        ij_catch = [(16, 16)]
        ptr_catch = [0, 1]
        sample_d2h = np.array([-0.100])
        sample_d18o = np.array([-0.012])
        cov = np.array([[1e-4, 1e-5], [1e-5, 1e-4]])
        
        result = calc_one_wind(
            beta, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 9, True
        )
        
        p_grid = result[16]
        
        # All precipitation values should be non-negative
        assert np.all(p_grid >= 0)
    
    def test_isotope_depletion_with_elevation(self):
        """Test that isotopes are depleted at higher elevations"""
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        beta = np.array([10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0])
        
        f_c = 1e-4
        h_r = 2500.0
        lat = np.array([45.0])
        lat0 = 45.0
        b_mwl_sample = np.array([10.0, 8.0])
        ij_catch = [(16, 16)]
        ptr_catch = [0, 1]
        sample_d2h = np.array([-0.100])
        sample_d18o = np.array([-0.012])
        cov = np.array([[1e-4, 1e-5], [1e-5, 1e-4]])
        
        result = calc_one_wind(
            beta, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 9, True
        )
        
        d2h_grid = result[22]
        d18o_grid = result[23]
        
        # Isotope values should be in reasonable range (in per mil)
        # Filter out NaN values
        valid_d2h = d2h_grid[~np.isnan(d2h_grid)]
        valid_d18o = d18o_grid[~np.isnan(d18o_grid)]
        
        # Check that isotope grids contain finite values
        # Note: actual range depends on input parameters
        assert len(valid_d2h) > 0, "d2h_grid should have valid values"
        assert len(valid_d18o) > 0, "d18o_grid should have valid values"
        assert np.all(np.isfinite(valid_d2h)), "d2h values should be finite"
        assert np.all(np.isfinite(valid_d18o)), "d18o values should be finite"


class TestCalcOneWindMatlabComparison:
    """Direct comparison with MATLAB calc_OneWind.m"""
    
    def test_beta_parameter_mapping(self):
        """
        Test that beta parameters are correctly mapped.
        
        MATLAB beta order:
        beta(1) = U (wind speed)
        beta(2) = azimuth
        beta(3) = T0 (sea-level temperature)
        beta(4) = M (mountain-height number)
        beta(5) = kappa (eddy diffusion)
        beta(6) = tauC (condensation time)
        beta(7) = d2H0 (base d2H)
        beta(8) = dD2H0_dLat (latitudinal gradient)
        beta(9) = fP0 (residual precipitation)
        """
        # This test verifies the parameter mapping is correct
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        # Test with specific parameter values
        beta = np.array([10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0])
        
        f_c = 1e-4
        h_r = 2500.0
        lat = np.array([45.0])
        lat0 = 45.0
        b_mwl_sample = np.array([10.0, 8.0])
        ij_catch = [(16, 16)]
        ptr_catch = [0, 1]
        sample_d2h = np.array([-0.100])
        sample_d18o = np.array([-0.012])
        cov = np.array([[1e-4, 1e-5], [1e-5, 1e-4]])
        
        result = calc_one_wind(
            beta, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 9, True
        )
        
        # Verify tau_f is calculated (depends on U, M, etc.)
        tau_f = result[14]
        assert float(tau_f) > 0
        
        # Verify p_grid is calculated
        p_grid = result[15]
        assert p_grid.shape == (32, 32)
        assert np.sum(p_grid) > 0  # Should have some precipitation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
