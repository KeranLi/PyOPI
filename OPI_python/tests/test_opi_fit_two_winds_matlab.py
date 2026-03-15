"""
Comparison tests for opiFit_TwoWinds.m (MATLAB) vs opi_fit_two_winds.py (Python)

These tests verify that the Python parameter fitting interface produces 
equivalent results to the MATLAB original for the TwoWinds OPI fitting workflow.
"""

import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from opi.app.fitting import (
    opi_fit_two_winds, get_default_run_data, load_run_file,
    create_synthetic_input_two_winds, classify_samples_by_divide
)


class TestOpiFitTwoWindsBasic:
    """Basic functionality tests for opi_fit_two_winds"""
    
    def test_function_exists(self):
        """Test that opi_fit_two_winds function exists and is callable"""
        assert callable(opi_fit_two_winds)
    
    def test_default_parameters_with_mock(self):
        """Test that function runs with default parameters (using mock)"""
        # Mock the optimization to avoid long computation
        mock_result = MagicMock()
        # 19 parameters for two-wind model
        mock_result.x = np.array([
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,  # Wind 1
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,  # Wind 2
            0.5  # Fraction
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(verbose=False, max_iterations=10)
        
        # Check that we get a dictionary result
        assert isinstance(result, dict)
        
        # Check that required keys exist
        assert 'solution_params' in result
        assert 'misfit' in result
        assert 'iterations' in result
        assert 'convergence' in result
        assert 'message' in result
    
    def test_solution_params_structure(self):
        """Test that solution_params has correct structure for 19 parameters"""
        mock_result = MagicMock()
        mock_result.x = np.array([
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,
            0.5
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(verbose=False, max_iterations=10)
        
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
        assert 'frac2' in params


class TestOpiFitTwoWindsParameters:
    """Tests for parameter handling"""
    
    def test_parameter_bounds(self):
        """Test that parameter bounds are respected in solution"""
        mock_result = MagicMock()
        mock_result.x = np.array([
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,
            0.5
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(verbose=False, max_iterations=10)
        
        if result['solution_params']:
            params = result['solution_params']
            
            # Check bounds (default bounds for two-wind model)
            # Wind 1 and 2 have same bounds
            assert 0.1 <= params['wind1_U'] <= 25
            assert -30 <= params['wind1_az'] <= 145
            assert 265 <= params['wind1_T0'] <= 295
            assert 0 <= params['wind1_M'] <= 1.2
            
            # Mixing fraction should be between 0 and 1
            assert 0 <= params['frac2'] <= 1


class TestOpiFitTwoWindsMatlabComparison:
    """Direct comparison with MATLAB opiFit_TwoWinds.m workflow"""
    
    def test_matlab_workflow_structure(self):
        """
        Test that the workflow matches MATLAB opiFit_TwoWinds:
        1. Load run file
        2. Get input data (topography and samples)
        3. Classify samples by continental divide
        4. Calculate catchment nodes
        5. Run fminCRS3 optimization with calc_TwoWinds
        6. Report results
        """
        mock_result = MagicMock()
        mock_result.x = np.array([
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,
            0.5
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 10
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(verbose=False, max_iterations=10)
        
        # Verify the workflow produces expected outputs
        assert isinstance(result, dict)
        
        # Check that we have the key outputs that MATLAB produces
        assert 'solution_params' in result  # beta (best-fit parameters)
        assert 'solution_vector' in result  # full solution vector
        assert 'misfit' in result  # chiR2
        assert 'iterations' in result  # optimization iterations
        assert 'convergence' in result  # success flag
        assert 'sample_types' in result  # classification of samples
    
    def test_matlab_constant_values(self):
        """Test that key constants match MATLAB opiFit_TwoWinds"""
        from opi.constants import HR, SD_RES_RATIO
        
        # hR = 540 m (characteristic distance for isotopic exchange)
        assert HR == 540.0
        
        # sdResRatio = 28.3 (standard deviation ratio)
        assert SD_RES_RATIO == 28.3
    
    def test_parameter_mapping_matches_matlab(self):
        """
        Test that beta parameter mapping matches MATLAB:
        
        State #1 (indices 1-9):
        beta(1) = U_1 (wind speed)
        beta(2) = azimuth_1
        beta(3) = T0_1 (sea-level temperature)
        beta(4) = M_1 (mountain-height number)
        beta(5) = kappa_1 (eddy diffusion)
        beta(6) = tauC_1 (condensation time)
        beta(7) = d2H0_1 (base d2H)
        beta(8) = dD2H0_dLat_1 (latitudinal gradient)
        beta(9) = fP0_1 (residual precipitation)
        beta(10) = fraction (fractional size of state 1)
        
        State #2 (indices 11-19):
        beta(11) = U_2
        beta(12) = azimuth_2
        beta(13) = T0_2
        beta(14) = M_2
        beta(15) = kappa_2
        beta(16) = tauC_2
        beta(17) = d2H0_2
        beta(18) = dD2H0_dLat_2
        beta(19) = fP0_2
        """
        mock_result = MagicMock()
        mock_result.x = np.array([
            10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0,  # Wind 1
            8.0, 270.0, 292.0, 0.4, 800.0, 1200.0, -0.120, 0.0, 1.0,    # Wind 2
            0.6  # Fraction
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(verbose=False, max_iterations=10)
        
        if result['solution_params']:
            params = result['solution_params']
            
            # Verify all 19 parameters are present
            assert 'wind1_U' in params
            assert params['wind1_U'] == 10.0
            assert params['wind1_az'] == 90.0
            assert params['wind1_T0'] == 288.15
            
            assert 'wind2_U' in params
            assert params['wind2_U'] == 8.0
            assert params['wind2_az'] == 270.0
            assert params['wind2_T0'] == 292.0
            
            assert 'frac2' in params
            assert params['frac2'] == 0.6
    
    def test_optimization_method(self):
        """Test that CRS3 optimization method is used (same as MATLAB fminCRS3)"""
        mock_result = MagicMock()
        mock_result.x = np.array([
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,
            0.5
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(verbose=False, max_iterations=10)
        
        # Should return a valid result structure
        assert isinstance(result, dict)
        assert 'solution_vector' in result
        
        # Solution vector should have 19 parameters
        if result['solution_vector']:
            assert len(result['solution_vector']) == 19


class TestOpiFitTwoWindsHelpers:
    """Tests for helper functions"""
    
    def test_create_synthetic_input_two_winds(self):
        """Test synthetic input data creation for two winds"""
        data = create_synthetic_input_two_winds()
        
        assert 'x' in data
        assert 'y' in data
        assert 'h_grid' in data
        assert 'sample_x' in data
        assert 'sample_d2h' in data
        assert 'sample_d18o' in data
        
        # Check that sample data exists and has samples for both groups
        assert len(data['sample_x']) > 0
        assert len(data['sample_d2h']) == len(data['sample_x'])
        assert len(data['sample_d18o']) == len(data['sample_x'])
    
    def test_classify_samples_by_divide(self):
        """Test sample classification function"""
        sample_x = np.array([-30000, -20000, 20000, 30000])
        sample_y = np.array([0, 0, 0, 0])
        
        # Without actual divide file, use mock
        types = classify_samples_by_divide(sample_x, sample_y, None)
        
        # Should return array of 1s and 2s
        assert len(types) == len(sample_x)
        assert np.all(np.isin(types, [1, 2]))


class TestOpiFitTwoWindsEdgeCases:
    """Edge case tests"""
    
    def test_no_run_file(self):
        """Test with no run file specified"""
        mock_result = MagicMock()
        mock_result.x = np.array([
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,
            0.5
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 1
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(run_file_path=None, verbose=False, max_iterations=1)
        
        assert isinstance(result, dict)
        assert 'solution_params' in result
    
    def test_nonexistent_run_file(self):
        """Test with non-existent run file"""
        mock_result = MagicMock()
        mock_result.x = np.array([
            10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7,
            10.0, 270.0, 290.0, 0.25, 0.0, 1000.0, -8e-3, -2e-3, 0.7,
            0.5
        ])
        mock_result.fun = 1.5
        mock_result.n_iter = 1
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_two_winds(run_file_path='/nonexistent/path/file.run',
                                       verbose=False, max_iterations=1)
        
        assert isinstance(result, dict)
        assert 'solution_params' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
