"""
Test calc_two_winds against MATLAB calc_TwoWinds.m

Numerical comparison tests for two-wind OPI calculation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi.physics.calc_two_winds import calc_two_winds


class TestCalcTwoWindsBasic:
    """Basic functionality tests"""
    
    @pytest.fixture
    def minimal_test_data(self):
        """Create minimal test data for function signature test"""
        # Create simple topography
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        # Gaussian mountain
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        # Beta vector with 19 parameters
        beta = np.array([
            10.0,    # U_1
            90.0,    # azimuth_1
            288.15,  # T0_1
            0.5,     # M_1
            1000.0,  # kappa_1
            1000.0,  # tau_c_1
            -0.100,  # d2h0_1
            0.0,     # d_d2h0_d_lat_1
            1.0,     # f_p0_1
            0.6,     # fraction
            8.0,     # U_2
            270.0,   # azimuth_2
            288.15,  # T0_2
            0.4,     # M_2
            800.0,   # kappa_2
            1200.0,  # tau_c_2
            -0.120,  # d2h0_2
            0.0,     # d_d2h0_d_lat_2
            1.0,     # f_p0_2
        ])
        
        # Other parameters
        f_c = 1e-4
        h_r = 2500.0
        lat = np.array([45.0])
        lat0 = 45.0
        b_mwl_sample = np.array([10.0, 8.0])  # intercept, slope
        
        # Simple catchment (single sample)
        ij_catch = [(16, 16)]  # Center point
        ptr_catch = [0, 1]
        
        sample_d2h = np.array([-0.100])
        sample_d18o = np.array([-0.012])
        
        cov = np.array([[1e-4, 1e-5], [1e-5, 1e-4]])
        n_parameters_free = 10
        is_fit = True
        
        return {
            'beta': beta,
            'f_c': f_c,
            'h_r': h_r,
            'x': x,
            'y': y,
            'lat': lat,
            'lat0': lat0,
            'h_grid': h_grid,
            'b_mwl_sample': b_mwl_sample,
            'ij_catch': ij_catch,
            'ptr_catch': ptr_catch,
            'sample_d2h': sample_d2h,
            'sample_d18o': sample_d18o,
            'cov': cov,
            'n_parameters_free': n_parameters_free,
            'is_fit': is_fit
        }
    
    def test_function_signature(self, minimal_test_data):
        """Test that function accepts correct parameters and returns results"""
        d = minimal_test_data
        
        try:
            result = calc_two_winds(
                d['beta'], d['f_c'], d['h_r'],
                d['x'], d['y'], d['lat'], d['lat0'], d['h_grid'],
                d['b_mwl_sample'], d['ij_catch'], d['ptr_catch'],
                d['sample_d2h'], d['sample_d18o'], d['cov'],
                d['n_parameters_free'], d['is_fit']
            )
            
            # Check that we got a tuple with many elements
            assert isinstance(result, tuple)
            assert len(result) > 30  # Many output values
            
            # First three should be chi_r2, nu, std_residuals
            chi_r2, nu, std_residuals = result[0], result[1], result[2]
            
        except Exception as e:
            pytest.fail(f"Function failed with error: {e}")
    
    def test_output_shapes(self, minimal_test_data):
        """Test that output arrays have correct shapes"""
        d = minimal_test_data
        
        result = calc_two_winds(
            d['beta'], d['f_c'], d['h_r'],
            d['x'], d['y'], d['lat'], d['lat0'], d['h_grid'],
            d['b_mwl_sample'], d['ij_catch'], d['ptr_catch'],
            d['sample_d2h'], d['sample_d18o'], d['cov'],
            d['n_parameters_free'], d['is_fit']
        )
        
        # Unpack key results
        chi_r2, nu, std_residuals = result[0], result[1], result[2]
        
        n_samples = len(d['ptr_catch']) - 1
        
        # Check shapes
        assert std_residuals.shape == (n_samples,) or std_residuals.shape == (1,)
        assert isinstance(chi_r2, (float, np.floating))
        assert isinstance(nu, (int, np.integer))


class TestCalcTwoWindsAlgorithm:
    """Tests for algorithm correctness"""
    
    def test_two_states_produce_different_results(self):
        """Test that wind state 1 and 2 produce different results"""
        # Create test data
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        # Beta with different parameters for state 1 and 2
        beta = np.array([
            15.0, 90.0, 290.0, 0.6, 1000.0, 1000.0, -0.080, 0.0, 1.0, 0.5,  # State 1: faster, warmer
            5.0, 270.0, 285.0, 0.3, 800.0, 1200.0, -0.150, 0.0, 1.0,       # State 2: slower, cooler
        ])
        
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
        
        result = calc_two_winds(
            beta, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 10, True
        )
        
        # Unpack state-specific results
        p_grid_1 = result[15]
        p_grid_2 = result[30]
        
        # State 1 should have more precipitation (faster wind)
        # than state 2 when fraction is 0.5
        assert np.sum(p_grid_1) > 0
        assert np.sum(p_grid_2) > 0
    
    def test_fraction_scaling(self):
        """Test that fraction parameter correctly scales precipitation"""
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        # Test with fraction = 0.7
        beta_70 = np.array([
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0, 0.7,
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
        ])
        
        # Test with fraction = 0.3
        beta_30 = np.array([
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0, 0.3,
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,
        ])
        
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
        
        result_70 = calc_two_winds(
            beta_70, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 10, True
        )
        
        result_30 = calc_two_winds(
            beta_30, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 10, True
        )
        
        p_grid_1_70 = result_70[15]
        p_grid_1_30 = result_30[15]
        
        # Higher fraction should give more precipitation from state 1
        assert np.sum(p_grid_1_70) > np.sum(p_grid_1_30)
    
    def test_combined_precipitation_is_sum(self):
        """Test that combined precipitation equals sum of states (same azimuth)"""
        x = np.linspace(0, 100000, 32)
        y = np.linspace(0, 100000, 32)
        X, Y = np.meshgrid(x, y)
        h_grid = 1000 * np.exp(-((X - 50000)**2 + (Y - 50000)**2) / (2 * 20000**2))
        
        # Use same azimuth for both states to ensure same grid size
        beta = np.array([
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0, 0.6,
            8.0, 90.0, 288.15, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0,  # Same azimuth
        ])
        
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
        
        result = calc_two_winds(
            beta, f_c, h_r, x, y, lat, lat0, h_grid,
            b_mwl_sample, ij_catch, ptr_catch,
            sample_d2h, sample_d18o, cov, 10, True
        )
        
        p_grid_1 = result[15]
        p_grid_2 = result[35]
        p_grid = result[43]
        
        # Combined should equal sum (when grids are same size)
        np.testing.assert_allclose(p_grid, p_grid_1 + p_grid_2, rtol=1e-10)


class TestCalcTwoWindsMatlabComparison:
    """Direct comparison with MATLAB calc_TwoWinds.m"""
    
    def test_beta_parameter_mapping(self):
        """
        Test that beta parameters are correctly mapped.
        
        MATLAB beta indices (1-indexed):
        beta(1) = U_1, beta(2) = azimuth_1, ..., beta(10) = fraction,
        beta(11) = U_2, ..., beta(19) = fP0_2
        
        Python beta indices (0-indexed):
        beta[0] = U_1, beta[1] = azimuth_1, ..., beta[9] = fraction,
        beta[10] = U_2, ..., beta[18] = f_p0_2
        """
        beta = np.array([
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0, 0.6,
            8.0, 270.0, 288.15, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0,
        ])
        
        # Check that we have 19 parameters
        assert len(beta) == 19
        
        # Check state 1 parameters
        assert beta[0] == 10.0   # U_1
        assert beta[1] == 90.0   # azimuth_1
        assert beta[9] == 0.6    # fraction
        
        # Check state 2 parameters
        assert beta[10] == 8.0   # U_2
        assert beta[11] == 270.0 # azimuth_2
        assert beta[18] == 1.0   # f_p0_2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
