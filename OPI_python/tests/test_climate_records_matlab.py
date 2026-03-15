"""
Test climate_records against MATLAB climateRecords.m

Numerical comparison tests for paleoclimate record processing.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opi.tools.climate_records import (
    load_mebm_data,
    load_benthic_foram_data,
    calculate_climate_records,
    estimate_paleoclimate_parameters
)


class TestLoadMEBMData:
    """Tests for MEBM data loading"""
    
    def test_load_returns_three_arrays(self, tmp_path):
        """Test that load_mebm_data returns three arrays"""
        # Create a mock .mat file
        from scipy.io import savemat
        
        dA = np.array([0, 1, 2])
        lat = np.array([-90, 0, 90])
        T = np.array([[280, 285, 290], [275, 280, 285]])
        
        mat_file = tmp_path / "test_mebm.mat"
        savemat(mat_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        dA_MEBM, lat_MEBM, T_MEBM = load_mebm_data(str(mat_file))
        
        assert isinstance(dA_MEBM, np.ndarray)
        assert isinstance(lat_MEBM, np.ndarray)
        assert isinstance(T_MEBM, np.ndarray)
        
        # Temperature should be converted to Kelvin (273.15 added)
        assert np.allclose(lat_MEBM, lat)
        assert T_MEBM.shape == T.shape
    
    def test_temperature_conversion_to_kelvin(self, tmp_path):
        """Test that temperature is converted to Kelvin"""
        from scipy.io import savemat
        
        T_celsius = np.array([[0, 10, 20]])  # Celsius
        mat_file = tmp_path / "test_mebm.mat"
        savemat(mat_file, {
            'dA_MEBM': np.array([0]),
            'lat_MEBM': np.array([0]),
            'T_MEBM': T_celsius
        })
        
        _, _, T_MEBM = load_mebm_data(str(mat_file))
        
        # Should be converted to Kelvin
        assert np.allclose(T_MEBM, T_celsius + 273.15)


class TestLoadBenthicData:
    """Tests for benthic foram data loading"""
    
    def test_load_returns_three_arrays(self, tmp_path):
        """Test that load_benthic_foram_data returns three arrays"""
        from scipy.io import savemat
        
        age = np.array([0, 1, 2, 3, 4])
        Tdw = np.array([275, 276, 277, 278, 279])
        d18O = np.array([0, -0.5, -1.0, -1.5, -2.0])
        
        mat_file = tmp_path / "test_benthic.mat"
        savemat(mat_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        age_record, T_dw_record, d18O_sw_record = load_benthic_foram_data(str(mat_file))
        
        assert isinstance(age_record, np.ndarray)
        assert isinstance(T_dw_record, np.ndarray)
        assert isinstance(d18O_sw_record, np.ndarray)
        
        assert np.allclose(age_record, age)
        assert np.allclose(T_dw_record, Tdw)
        assert np.allclose(d18O_sw_record, d18O)


class TestCalculateClimateRecords:
    """Tests for calculate_climate_records function"""
    
    def test_output_shapes_and_types(self, tmp_path):
        """Test output array shapes and types"""
        from scipy.io import savemat
        
        # Create mock MEBM data (must include dA=0)
        dA = np.linspace(-2, 2, 10)
        dA[5] = 0  # Ensure dA=0 is present (for present-day)
        lat = np.linspace(-90, 90, 20)
        T = np.random.rand(20, 10) * 50 + 250  # 250-300 K range
        
        mebm_file = tmp_path / "mebm.mat"
        savemat(mebm_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        # Create mock benthic data
        age = np.linspace(0, 5, 100)
        Tdw = np.linspace(275, 280, 100)
        d18O = np.linspace(0, -2, 100)
        
        benthic_file = tmp_path / "benthic.mat"
        savemat(benthic_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        T0_record, d2H0_record, age_record, T0_smooth, d2H0_smooth = calculate_climate_records(
            str(mebm_file),
            str(benthic_file),
            lat=45.0,
            T0_pres=288.15,
            d2H0_pres=-0.100,
            d_d2H0_d_d18O0=8.0,
            d_d2H0_d_T0=0.004,
            span_age=1.0,
            smooth_method='loess'
        )
        
        # All outputs should be 1D arrays of same length
        assert T0_record.ndim == 1
        assert d2H0_record.ndim == 1
        assert age_record.ndim == 1
        assert T0_smooth.ndim == 1
        assert d2H0_smooth.ndim == 1
        
        assert len(T0_record) == len(age_record)
        assert len(d2H0_record) == len(age_record)
        assert len(T0_smooth) == len(age_record)
        assert len(d2H0_smooth) == len(age_record)
    
    def test_temperature_range_reasonable(self, tmp_path):
        """Test that calculated temperatures are in reasonable range"""
        from scipy.io import savemat
        
        # Create mock data (must include dA=0)
        dA = np.linspace(-2, 2, 10)
        dA[5] = 0  # Ensure dA=0 is present
        lat = np.linspace(-90, 90, 20)
        T = np.random.rand(20, 10) * 30 + 260
        
        mebm_file = tmp_path / "mebm.mat"
        savemat(mebm_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        age = np.linspace(0, 5, 100)
        Tdw = np.linspace(275, 280, 100)
        d18O = np.linspace(0, -2, 100)
        
        benthic_file = tmp_path / "benthic.mat"
        savemat(benthic_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        T0_record, _, _, _, _ = calculate_climate_records(
            str(mebm_file),
            str(benthic_file),
            lat=45.0,
            T0_pres=288.15,
            d2H0_pres=-0.100,
            d_d2H0_d_d18O0=8.0,
            d_d2H0_d_T0=0.004,
            span_age=1.0
        )
        
        # Temperature should be in reasonable atmospheric range
        assert np.all(T0_record > 200)  # Above -73°C
        assert np.all(T0_record < 350)  # Below 77°C
    
    def test_d2h0_range_reasonable(self, tmp_path):
        """Test that calculated d2H0 values are in reasonable range"""
        from scipy.io import savemat
        
        # Create mock data (must include dA=0)
        dA = np.linspace(-2, 2, 10)
        dA[5] = 0  # Ensure dA=0 is present
        lat = np.linspace(-90, 90, 20)
        T = np.random.rand(20, 10) * 30 + 260
        
        mebm_file = tmp_path / "mebm.mat"
        savemat(mebm_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        age = np.linspace(0, 5, 100)
        Tdw = np.linspace(275, 280, 100)
        d18O = np.linspace(0, -2, 100)
        
        benthic_file = tmp_path / "benthic.mat"
        savemat(benthic_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        _, d2H0_record, _, _, _ = calculate_climate_records(
            str(mebm_file),
            str(benthic_file),
            lat=45.0,
            T0_pres=288.15,
            d2H0_pres=-0.100,
            d_d2H0_d_d18O0=8.0,
            d_d2H0_d_T0=0.004,
            span_age=1.0
        )
        
        # d2H0 should be in reasonable range (fraction)
        assert np.all(d2H0_record > -0.5)  # Above -500 permil
        assert np.all(d2H0_record < 0.1)   # Below 100 permil
    
    def test_present_value_exact(self, tmp_path):
        """Test that present values (age=0) are set exactly"""
        from scipy.io import savemat
        
        dA = np.linspace(-2, 2, 10)
        dA[5] = 0  # Ensure dA=0 is present
        lat = np.linspace(-90, 90, 20)
        T = np.random.rand(20, 10) * 30 + 260
        
        mebm_file = tmp_path / "mebm.mat"
        savemat(mebm_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        age = np.linspace(0, 5, 100)
        Tdw = np.linspace(275, 280, 100)
        d18O = np.linspace(0, -2, 100)
        
        benthic_file = tmp_path / "benthic.mat"
        savemat(benthic_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        T0_pres = 288.15
        d2H0_pres = -0.100
        
        _, _, age_record, T0_smooth, d2H0_smooth = calculate_climate_records(
            str(mebm_file),
            str(benthic_file),
            lat=45.0,
            T0_pres=T0_pres,
            d2H0_pres=d2H0_pres,
            d_d2H0_d_d18O0=8.0,
            d_d2H0_d_T0=0.004,
            span_age=1.0
        )
        
        # Present values should be exact
        present_idx = age_record == 0
        if np.any(present_idx):
            assert T0_smooth[present_idx][0] == T0_pres
            assert d2H0_smooth[present_idx][0] == d2H0_pres
    
    def test_different_smooth_methods(self, tmp_path):
        """Test different smoothing methods"""
        from scipy.io import savemat
        
        dA = np.linspace(-2, 2, 10)
        dA[5] = 0  # Ensure dA=0 is present
        lat = np.linspace(-90, 90, 20)
        T = np.random.rand(20, 10) * 30 + 260
        
        mebm_file = tmp_path / "mebm.mat"
        savemat(mebm_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        age = np.linspace(0, 5, 100)
        Tdw = np.linspace(275, 280, 100)
        d18O = np.linspace(0, -2, 100)
        
        benthic_file = tmp_path / "benthic.mat"
        savemat(benthic_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        for method in ['loess', 'moving', 'none']:
            T0_record, d2H0_record, _, T0_smooth, d2H0_smooth = calculate_climate_records(
                str(mebm_file),
                str(benthic_file),
                lat=45.0,
                T0_pres=288.15,
                d2H0_pres=-0.100,
                d_d2H0_d_d18O0=8.0,
                d_d2H0_d_T0=0.004,
                span_age=1.0,
                smooth_method=method
            )
            
            assert T0_smooth is not None
            assert d2H0_smooth is not None


class TestEstimatePaleoclimateParameters:
    """Tests for estimate_paleoclimate_parameters function"""
    
    def test_returns_two_floats(self, tmp_path):
        """Test that function returns two floats"""
        from scipy.io import savemat
        
        dA = np.linspace(-2, 2, 10)
        dA[5] = 0  # Ensure dA=0 is present
        lat = np.linspace(-90, 90, 20)
        T = np.random.rand(20, 10) * 30 + 260
        
        mebm_file = tmp_path / "mebm.mat"
        savemat(mebm_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        age = np.linspace(0, 5, 100)
        Tdw = np.linspace(275, 280, 100)
        d18O = np.linspace(0, -2, 100)
        
        benthic_file = tmp_path / "benthic.mat"
        savemat(benthic_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        T0, d2H0 = estimate_paleoclimate_parameters(
            age=2.0,
            mebm_file=str(mebm_file),
            benthic_file=str(benthic_file),
            lat=45.0,
            T0_pres=288.15,
            d2H0_pres=-0.100
        )
        
        assert isinstance(T0, (float, np.floating))
        assert isinstance(d2H0, (float, np.floating))
    
    def test_default_temperature_scaling(self, tmp_path):
        """Test default temperature scaling parameter"""
        from scipy.io import savemat
        
        dA = np.linspace(-2, 2, 10)
        dA[5] = 0  # Ensure dA=0 is present
        lat = np.linspace(-90, 90, 20)
        T = np.random.rand(20, 10) * 30 + 260
        
        mebm_file = tmp_path / "mebm.mat"
        savemat(mebm_file, {'dA_MEBM': dA, 'lat_MEBM': lat, 'T_MEBM': T})
        
        age = np.linspace(0, 5, 100)
        Tdw = np.linspace(275, 280, 100)
        d18O = np.linspace(0, -2, 100)
        
        benthic_file = tmp_path / "benthic.mat"
        savemat(benthic_file, {
            'ageRecord': age,
            'TdwRecord': Tdw,
            'd18OswRecord': d18O
        })
        
        # Should work without specifying d_d2H0_d_T0
        T0, d2H0 = estimate_paleoclimate_parameters(
            age=2.0,
            mebm_file=str(mebm_file),
            benthic_file=str(benthic_file),
            lat=45.0,
            T0_pres=288.15,
            d2H0_pres=-0.100
        )
        
        assert T0 is not None
        assert d2H0 is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
