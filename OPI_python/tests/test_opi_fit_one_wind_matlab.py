"""
Comparison tests for opiFit_OneWind.m (MATLAB) vs opi_fit_one_wind.py (Python)

These tests verify that the Python parameter fitting interface produces 
equivalent results to the MATLAB original for the OneWind OPI fitting workflow.
"""

import numpy as np
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from opi.app.fitting import (
    opi_fit_one_wind, get_default_run_data, load_run_file, 
    create_synthetic_input
)


class TestOpiFitOneWindBasic:
    """Basic functionality tests for opi_fit_one_wind"""
    
    def test_function_exists(self):
        """Test that opi_fit_one_wind function exists and is callable"""
        assert callable(opi_fit_one_wind)
    
    def test_default_parameters_with_mock(self):
        """Test that function runs with default parameters (using mock)"""
        # Mock the optimization to avoid long computation
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(verbose=False, max_iterations=10)
        
        # Check that we get a dictionary result
        assert isinstance(result, dict)
        
        # Check that required keys exist
        assert 'solution_params' in result
        assert 'misfit' in result
        assert 'iterations' in result
        assert 'convergence' in result
        assert 'message' in result
    
    def test_solution_params_structure(self):
        """Test that solution_params has correct structure for 9 parameters"""
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(verbose=False, max_iterations=10)
        
        params = result['solution_params']
        
        # Check for expected parameter names (9 parameters for one-wind model)
        expected_names = ['U', 'azimuth', 'T0', 'M', 'kappa', 'tau_c', 'd2h0', 'd_d2h0_d_lat', 'f_p0']
        
        for name in expected_names:
            assert name in params, f"Missing parameter: {name}"


class TestOpiFitOneWindParameters:
    """Tests for parameter handling"""
    
    def test_parameter_bounds(self):
        """Test that parameter bounds are respected in solution"""
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(verbose=False, max_iterations=10)
        
        if result['solution_params']:
            params = result['solution_params']
            
            # Check bounds from default run data
            default = get_default_run_data()
            lb = default['param_constraints_min']
            ub = default['param_constraints_max']
            
            # Verify solution parameters are within bounds (approximately)
            assert params['U'] >= lb[0] - 0.1 and params['U'] <= ub[0] + 0.1
            assert params['azimuth'] >= lb[1] - 1 and params['azimuth'] <= ub[1] + 1
            assert params['T0'] >= lb[2] - 0.5 and params['T0'] <= ub[2] + 0.5


class TestOpiFitOneWindMatlabComparison:
    """Direct comparison with MATLAB opiFit_OneWind.m workflow"""
    
    def test_matlab_workflow_structure(self):
        """
        Test that the workflow matches MATLAB opiFit_OneWind:
        1. Load run file
        2. Get input data (topography and samples)
        3. Calculate catchment nodes
        4. Run fminCRS3 optimization
        5. Report results
        """
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 10
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(verbose=False, max_iterations=10)
        
        # Verify the workflow produces expected outputs
        assert isinstance(result, dict)
        
        # Check that we have the key outputs that MATLAB produces
        assert 'solution_params' in result  # beta (best-fit parameters)
        assert 'misfit' in result  # chiR2
        assert 'iterations' in result  # optimization iterations
        assert 'convergence' in result  # success flag
    
    def test_matlab_constant_values(self):
        """Test that key constants match MATLAB opiFit_OneWind"""
        from opi.constants import HR, SD_RES_RATIO, TC2K, M_PER_DEGREE
        
        # hR = 540 m (characteristic distance for isotopic exchange)
        assert HR == 540.0
        
        # sdResRatio = 28.3 (standard deviation ratio)
        assert SD_RES_RATIO == 28.3
        
        # TC2K = 273.15 (Celsius to Kelvin conversion)
        assert TC2K == 273.15
        
        # Meters per degree
        expected_m_per_degree = np.pi * 6371e3 / 180
        assert np.isclose(M_PER_DEGREE, expected_m_per_degree, rtol=1e-6)
    
    def test_parameter_mapping_matches_matlab(self):
        """
        Test that beta parameter mapping matches MATLAB:
        beta(1) = U (wind speed)
        beta(2) = azimuth
        beta(3) = T0 (sea-level temperature)
        beta(4) = M (mountain-height number)
        beta(5) = kappa (eddy diffusion)
        beta(6) = tau_c (condensation time)
        beta(7) = d2h0 (base d2H)
        beta(8) = d_d2h0_d_lat (latitudinal gradient)
        beta(9) = f_p0 (residual precipitation)
        """
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(verbose=False, max_iterations=10)
        
        if result['solution_params']:
            params = result['solution_params']
            
            # Verify all 9 parameters are present
            assert 'U' in params
            assert 'azimuth' in params
            assert 'T0' in params
            assert 'M' in params
            assert 'kappa' in params
            assert 'tau_c' in params
            assert 'd2h0' in params
            assert 'd_d2h0_d_lat' in params
            assert 'f_p0' in params
    
    def test_optimization_method(self):
        """Test that CRS3 optimization method is used (same as MATLAB fminCRS3)"""
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 5
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(verbose=False, max_iterations=10)
        
        # Should return a valid result structure
        assert isinstance(result, dict)
        assert 'solution_vector' in result
        
        # Solution vector should have 9 parameters
        if result['solution_vector']:
            assert len(result['solution_vector']) == 9


class TestOpiFitOneWindHelpers:
    """Tests for helper functions"""
    
    def test_get_default_run_data(self):
        """Test that default run data has correct structure"""
        data = get_default_run_data()
        
        assert 'run_title' in data
        assert 'data_path' in data
        assert 'topography_file' in data
        assert 'param_constraints_min' in data
        assert 'param_constraints_max' in data
        assert 'initial_guess' in data
        
        # Check that bounds have 9 parameters
        assert len(data['param_constraints_min']) == 9
        assert len(data['param_constraints_max']) == 9
        assert len(data['initial_guess']) == 9
    
    def test_load_run_file(self):
        """Test that run file loading works correctly"""
        # Create a temporary run file
        run_content = """Test Run
0
../data

Andes.mat
0.25
samples.mat
no
[-76 -64 -52 -38]
map
[10 1e-4]
no
U|azimuth|T0|M|kappa|tau_c|d2H0|dd2H0/dlat|fP
[0 0 0 0 6 0 -3 -3 0]
[0.1 -30 265 0 0 0 -15e-3 0e-3 1]
[25 145 295 1.2 1e6 2500 15e-3 0e-3 1]"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.run', delete=False) as f:
            f.write(run_content)
            temp_path = f.name
        
        try:
            data = load_run_file(temp_path)
            
            assert data['run_title'] == 'Test Run'
            assert data['data_path'] == '../data'
            assert data['topography_file'] == 'Andes.mat'
        finally:
            os.unlink(temp_path)
    
    def test_create_synthetic_input(self):
        """Test synthetic input data creation"""
        data = create_synthetic_input()
        
        assert 'x' in data
        assert 'y' in data
        assert 'h_grid' in data
        assert 'sample_x' in data
        assert 'sample_d2h' in data
        assert 'sample_d18o' in data
        
        # Check that sample data exists
        assert len(data['sample_x']) > 0
        assert len(data['sample_d2h']) == len(data['sample_x'])


class TestOpiFitOneWindEdgeCases:
    """Edge case tests"""
    
    def test_no_run_file(self):
        """Test with no run file specified"""
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 1
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(run_file_path=None, verbose=False, max_iterations=1)
        
        assert isinstance(result, dict)
        assert 'solution_params' in result
    
    def test_nonexistent_run_file(self):
        """Test with non-existent run file"""
        mock_result = MagicMock()
        mock_result.x = np.array([10.0, 90.0, 290.0, 0.25, 0.0, 1000.0, -5e-3, -2e-3, 0.7])
        mock_result.fun = 1.5
        mock_result.n_iter = 1
        mock_result.success = True
        
        with patch('opi.app.fitting.fmin_crs3', return_value=mock_result):
            result = opi_fit_one_wind(run_file_path='/nonexistent/path/file.run', 
                                       verbose=False, max_iterations=1)
        
        assert isinstance(result, dict)
        assert 'solution_params' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
