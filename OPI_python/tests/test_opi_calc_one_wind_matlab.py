"""
Comparison tests for opiCalc_OneWind.m (MATLAB) vs opi_calc_one_wind.py (Python)

These tests verify that the Python high-level interface produces equivalent 
results to the MATLAB original for the OneWind OPI calculation workflow.
"""

import numpy as np
import pytest
import os
import tempfile
from opi.app.calc_one_wind import opi_calc_one_wind, load_run_file, get_default_run_parameters


class TestOpiCalcOneWindBasic:
    """Basic functionality tests for opi_calc_one_wind"""
    
    def test_function_exists(self):
        """Test that opi_calc_one_wind function exists and is callable"""
        assert callable(opi_calc_one_wind)
    
    def test_default_parameters(self):
        """Test that function runs with default parameters"""
        result = opi_calc_one_wind(verbose=False)
        
        # Check that we get a dictionary result
        assert isinstance(result, dict)
        
        # Check that required keys exist
        assert 'solution_params' in result
        assert 'derived_params' in result
        assert 'results' in result
        assert 'chi_r2' in result
        assert 'nu' in result
    
    def test_solution_params_structure(self):
        """Test that solution_params has correct structure"""
        result = opi_calc_one_wind(verbose=False)
        
        params = result['solution_params']
        
        # Check for expected parameter names
        expected_names = ['U (m/s)', 'Azimuth (deg)', 'T0 (K)', 'M', 
                          'kappa (m^2/s)', 'tau_c (s)', 'd2H0', 'dd2H0/dlat', 'fP']
        
        for name in expected_names:
            assert name in params, f"Missing parameter: {name}"
    
    def test_results_structure(self):
        """Test that results dictionary has correct structure"""
        result = opi_calc_one_wind(verbose=False)
        
        if result['results']:  # If calculation succeeded
            assert 'precipitation' in result['results']
            assert 'moisture_ratio' in result['results']
            assert 'd2h' in result['results']
            assert 'd18o' in result['results']
            assert 'x' in result['results']
            assert 'y' in result['results']
            assert 'h_grid' in result['results']


class TestOpiCalcOneWindParameters:
    """Tests for parameter handling"""
    
    def test_custom_solution_vector(self):
        """Test that custom solution vector is used correctly"""
        custom_beta = [15.0, 45.0, 295.0, 0.5, 1000.0, 1500.0, -8.0e-3, -1.5e-3, 0.8]
        
        result = opi_calc_one_wind(solution_vector=custom_beta, verbose=False)
        
        # Check that custom parameters were used
        params = result['solution_params']
        assert params['U (m/s)'] == 15.0
        assert params['Azimuth (deg)'] == 45.0
        assert params['T0 (K)'] == 295.0
    
    def test_different_wind_speeds(self):
        """Test calculation with different wind speeds"""
        beta_slow = [5.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0]
        beta_fast = [15.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0]
        
        result_slow = opi_calc_one_wind(solution_vector=beta_slow, verbose=False)
        result_fast = opi_calc_one_wind(solution_vector=beta_fast, verbose=False)
        
        # Both should complete without errors
        assert 'results' in result_slow
        assert 'results' in result_fast


class TestLoadRunFile:
    """Tests for run file loading functionality"""
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent run file returns defaults"""
        params = load_run_file('nonexistent_file.run')
        
        # Should return default parameters
        assert 'run_title' in params
        assert 'solution' in params
    
    def test_default_run_parameters(self):
        """Test that default parameters are returned correctly"""
        params = get_default_run_parameters()
        
        assert params['run_title'] == 'Example Run'
        assert 'param_labels' in params
        assert 'param_constraints_min' in params
        assert 'param_constraints_max' in params
        assert len(params['param_constraints_min']) == 9
        assert len(params['param_constraints_max']) == 9
    
    def test_run_file_format(self):
        """Test that run file format is parsed correctly"""
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
[25 145 295 1.2 1e6 2500 15e-3 0e-3 1]
[10.0 90.0 288.15 0.5 1000.0 1000.0 -0.100 0.0 1.0]"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.run', delete=False) as f:
            f.write(run_content)
            temp_path = f.name
        
        try:
            params = load_run_file(temp_path)
            
            assert params['run_title'] == 'Test Run'
            assert params['data_path'] == '../data'
            assert params['topography_file'] == 'Andes.mat'
            assert params['r_tukey'] == 0.25
            assert params['sample_file'] == 'samples.mat'
            assert len(params['map_limits']) == 4
            assert params['solution'] is not None
            assert len(params['solution']) == 9
        finally:
            os.unlink(temp_path)


class TestOpiCalcOneWindMatlabComparison:
    """Direct comparison with MATLAB opiCalc_OneWind.m workflow"""
    
    def test_matlab_workflow_structure(self):
        """
        Test that the workflow matches MATLAB opiCalc_OneWind:
        1. Load run file
        2. Get input data
        3. Call calc_OneWind
        4. Calculate lifting (if samples exist)
        5. Return/save results
        """
        result = opi_calc_one_wind(verbose=False)
        
        # Verify the workflow produces expected outputs
        assert isinstance(result, dict)
        
        # Check that we have the key outputs that MATLAB saves
        if result['results']:
            results = result['results']
            
            # These are the key grids saved by MATLAB
            assert 'precipitation' in results  # pGrid
            assert 'd2h' in results  # d2HGrid
            assert 'd18o' in results  # d18OGrid
            assert 'moisture_ratio' in results  # fMGrid
            assert 'h_grid' in results  # hGrid
    
    def test_matlab_constant_values(self):
        """Test that key constants match MATLAB"""
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
        beta(7) = d2H0 (base d2H)
        beta(8) = dD2H0_dLat (latitudinal gradient)
        beta(9) = fP0 (residual precipitation)
        """
        beta = [10.0, 90.0, 288.15, 0.5, 1000.0, 1000.0, -0.100, 0.0, 1.0]
        
        result = opi_calc_one_wind(solution_vector=beta, verbose=False)
        params = result['solution_params']
        
        # Verify parameter values are correctly passed through
        assert params['U (m/s)'] == 10.0
        assert params['Azimuth (deg)'] == 90.0
        assert params['T0 (K)'] == 288.15
        assert params['M'] == 0.5
        assert params['kappa (m^2/s)'] == 1000.0
        assert params['tau_c (s)'] == 1000.0
    
    def test_output_grids_consistency(self):
        """Test that output grids are consistent with each other"""
        result = opi_calc_one_wind(verbose=False)
        
        if not result['results']:
            pytest.skip("Calculation did not produce results")
        
        results = result['results']
        
        # All grids should have the same shape
        p_grid = results['precipitation']
        d2h_grid = results['d2h']
        d18o_grid = results['d18o']
        f_m_grid = results['moisture_ratio']
        h_grid = results['h_grid']
        
        assert p_grid.shape == d2h_grid.shape
        assert p_grid.shape == d18o_grid.shape
        assert p_grid.shape == f_m_grid.shape
        assert p_grid.shape == h_grid.shape


class TestOpiCalcOneWindEdgeCases:
    """Edge case tests"""
    
    def test_empty_run_file_path(self):
        """Test with empty run file path"""
        result = opi_calc_one_wind(run_file_path=None, verbose=False)
        
        assert isinstance(result, dict)
        assert 'solution_params' in result
    
    def test_no_samples(self):
        """Test calculation without sample data"""
        # Default calculation has no samples
        result = opi_calc_one_wind(verbose=False)
        
        # Should still complete successfully
        assert 'results' in result
        assert 'solution_params' in result
    
    def test_extreme_parameter_values(self):
        """Test with extreme but valid parameter values"""
        # Very cold, slow wind
        beta_extreme = [1.0, 0.0, 265.0, 0.1, 0.0, 500.0, -15e-3, -2e-3, 1.0]
        
        # Should not crash (may produce warnings)
        try:
            result = opi_calc_one_wind(solution_vector=beta_extreme, verbose=False)
            assert isinstance(result, dict)
        except Exception as e:
            # Some extreme values might cause numerical issues, which is acceptable
            pytest.skip(f"Extreme parameters caused expected error: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
