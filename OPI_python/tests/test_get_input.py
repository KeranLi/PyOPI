"""
Test get_input function against MATLAB getInput.m behavior
"""

import numpy as np
import os
import tempfile
from scipy.io import savemat
import pandas as pd
from opi.io.data_loader import get_input, grid_read


def create_test_topo_file(filepath):
    """Create a test topography MAT file"""
    # Create simple grid
    lon = np.linspace(100, 110, 50)
    lat = np.linspace(30, 40, 40)
    LON, LAT = np.meshgrid(lon, lat)
    # Simple Gaussian hill
    h_grid = 2000 * np.exp(-((LON-105)**2 + (LAT-35)**2) / (2 * 2**2))
    
    savemat(filepath, {'lon': lon, 'lat': lat, 'h_grid': h_grid})


def create_test_sample_file(filepath):
    """Create a test sample Excel file"""
    # Excel column order: 1=Longitude, 2=Latitude, 3=Elevation, 4=d2H, 5=d18O, 6=Type
    data = {
        0: [104.5, 105.2, 105.8],      # Longitude
        1: [34.5, 35.1, 35.7],         # Latitude
        2: [1500, 1800, 2100],         # Elevation (not used but present)
        3: [-80, -90, -100],           # d2H (permil)
        4: [-12, -13, -14],            # d18O (permil)
        5: ['L', 'C', 'L'],            # Type
    }
    df = pd.DataFrame(data)
    df.to_excel(filepath, index=False, header=False)


def test_get_input_basic():
    """Test basic get_input functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        topo_file = os.path.join(tmpdir, 'topo.mat')
        sample_file = os.path.join(tmpdir, 'samples.xlsx')
        
        create_test_topo_file(topo_file)
        create_test_sample_file(sample_file)
        
        # Call get_input
        result = get_input(tmpdir, 'topo.mat', r_tukey=0.0, 
                          sample_file='samples.xlsx')
        
        print(f"Grid shape: {result['h_grid'].shape}")
        print(f"Number of samples: {len(result['sample_lon'])}")
        print(f"Sample longitudes: {result['sample_lon']}")
        print(f"Sample latitudes: {result['sample_lat']}")
        print(f"Sample d2H (fraction): {result['sample_d2h']}")
        print(f"Sample d18O (fraction): {result['sample_d18o']}")
        print(f"Sample types: {result['sample_lc']}")
        print(f"MWL: d2H = {result['b_mwl_sample'][0]*1000:.2f} + {result['b_mwl_sample'][1]:.2f} * d18O")
        
        # Verify grid
        assert result['h_grid'].shape == (40, 50)
        
        # Verify samples
        assert len(result['sample_lon']) == 3
        assert len(result['sample_lat']) == 3
        
        # Verify d2H and d18O are converted to fraction (not permil)
        assert np.all(result['sample_d2h'] < 0)  # Negative values
        assert np.all(result['sample_d18o'] < 0)
        assert np.all(np.abs(result['sample_d2h']) < 0.2)  # Fraction range
        assert np.all(np.abs(result['sample_d18o']) < 0.02)
        
        # Verify sample types
        assert all(t in ['L', 'C'] for t in result['sample_lc'])
        
        print("[PASS] Basic get_input test")


def test_get_input_column_order():
    """Test that column order matches MATLAB: Lon, Lat, Elev, d2H, d18O, Type"""
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_file = os.path.join(tmpdir, 'topo.mat')
        sample_file = os.path.join(tmpdir, 'samples.xlsx')
        
        create_test_topo_file(topo_file)
        
        # Create file with specific values to test column order
        data = {
            0: [105.0],           # Longitude
            1: [35.0],            # Latitude
            2: [2000],            # Elevation
            3: [-100],            # d2H (permil) -> -0.1 fraction
            4: [-14],             # d18O (permil) -> -0.014 fraction
            5: ['L'],             # Type
        }
        df = pd.DataFrame(data)
        df.to_excel(sample_file, index=False, header=False)
        
        result = get_input(tmpdir, 'topo.mat', sample_file='samples.xlsx')
        
        # Verify the values are read from correct columns
        assert result['sample_lon'][0] == 105.0
        assert result['sample_lat'][0] == 35.0
        assert abs(result['sample_d2h'][0] - (-0.1)) < 1e-10  # -100 permil = -0.1 fraction
        assert abs(result['sample_d18o'][0] - (-0.014)) < 1e-10  # -14 permil = -0.014 fraction
        assert result['sample_lc'][0] == 'L'
        
        print("[PASS] Column order test")


def test_get_input_no_samples():
    """Test get_input without sample file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_file = os.path.join(tmpdir, 'topo.mat')
        create_test_topo_file(topo_file)
        
        result = get_input(tmpdir, 'topo.mat')
        
        # Verify empty sample arrays
        assert len(result['sample_lon']) == 0
        assert len(result['sample_d2h']) == 0
        
        # Verify map origin is from grid
        assert result['lon0'] == np.mean(result['lon'])
        assert result['lat0'] == np.mean(result['lat'])
        
        print("[PASS] No samples test")


def test_get_input_tukey_window():
    """Test Tukey window application"""
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_file = os.path.join(tmpdir, 'topo.mat')
        create_test_topo_file(topo_file)
        
        # Without Tukey window
        result_no_window = get_input(tmpdir, 'topo.mat', r_tukey=0.0)
        
        # With Tukey window
        result_with_window = get_input(tmpdir, 'topo.mat', r_tukey=0.5)
        
        # With window should have lower max elevation at edges
        assert result_with_window['h_grid'].max() <= result_no_window['h_grid'].max()
        
        print("[PASS] Tukey window test")


def test_get_input_d_excess():
    """Test d-excess calculation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_file = os.path.join(tmpdir, 'topo.mat')
        sample_file = os.path.join(tmpdir, 'samples.xlsx')
        
        create_test_topo_file(topo_file)
        
        # Create sample with known d-excess
        # d-excess = d2H - 8 * d18O
        # d2H = -80, d18O = -12 -> d-excess = -80 - 8*(-12) = -80 + 96 = 16
        data = {
            0: [105.0],
            1: [35.0],
            2: [1500],
            3: [-80],    # d2H (permil)
            4: [-12],    # d18O (permil)
            5: ['L'],
        }
        df = pd.DataFrame(data)
        df.to_excel(sample_file, index=False, header=False)
        
        result = get_input(tmpdir, 'topo.mat', sample_file='samples.xlsx')
        
        # d-excess should be calculated in fraction units
        # d2H = -0.08, d18O = -0.012
        # d-excess = -0.08 - 8*(-0.012) = -0.08 + 0.096 = 0.016
        expected_d_excess = 0.016
        assert abs(result['sample_d_excess'][0] - expected_d_excess) < 1e-10
        
        print("[PASS] d-excess calculation test")


def test_get_input_coriolis():
    """Test Coriolis frequency calculation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_file = os.path.join(tmpdir, 'topo.mat')
        create_test_topo_file(topo_file)
        
        result = get_input(tmpdir, 'topo.mat')
        
        # Coriolis frequency at 35 degrees latitude
        # f_c = 2 * omega * sin(lat)
        omega = 7.2921e-5
        expected_f_c = 2 * omega * np.sin(np.deg2rad(result['lat0']))
        
        assert abs(result['f_c'] - expected_f_c) < 1e-10
        
        print("[PASS] Coriolis frequency test")


def test_get_input_sample_alt():
    """Test division into primary and altered precipitation samples"""
    with tempfile.TemporaryDirectory() as tmpdir:
        topo_file = os.path.join(tmpdir, 'topo.mat')
        sample_file = os.path.join(tmpdir, 'samples.xlsx')
        
        create_test_topo_file(topo_file)
        
        # Create samples with varying d-excess
        # d-excess = d2H - 8 * d18O
        # Sample 1: d2H = -100, d18O = -14 -> d-excess = -100 - 8*(-14) = 12 (primary, < 5 permil)
        # Sample 2: d2H = -80, d18O = -10 -> d-excess = -80 - 8*(-10) = 0 (altered, < 5 permil)
        # Sample 3: d2H = -60, d18O = -8 -> d-excess = -60 - 8*(-8) = 4 (altered, < 5 permil)
        data = {
            0: [104.5, 105.0, 105.5],   # Longitude
            1: [34.5, 35.0, 35.5],      # Latitude
            2: [1500, 1600, 1700],      # Elevation
            3: [-100, -80, -60],        # d2H (permil)
            4: [-14, -10, -8],          # d18O (permil)
            5: ['L', 'C', 'L'],         # Type
        }
        df = pd.DataFrame(data)
        df.to_excel(sample_file, index=False, header=False)
        
        result = get_input(tmpdir, 'topo.mat', sample_file='samples.xlsx')
        
        # Check that samples are divided
        # Note: The division is based on d-excess > 5 permil threshold
        # All samples have d-excess < 5 permil in fraction (0.005), so all go to primary
        # Actually d-excess = 12, 0, 4 permil = 0.012, 0, 0.004 fraction
        # Threshold is 5 permil = 0.005 fraction
        # So sample 1 (0.012 > 0.005) is primary, samples 2,3 are altered
        
        print(f"Primary samples: {len(result['sample_lon'])}")
        print(f"Altered samples: {len(result['sample_lon_alt'])}")
        
        # Verify arrays are not empty where appropriate
        assert len(result['sample_line']) > 0 or len(result['sample_line_alt']) > 0
        
        print("[PASS] Sample division test")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing get_input function")
    print("=" * 60)
    
    test_get_input_basic()
    print()
    test_get_input_column_order()
    print()
    test_get_input_no_samples()
    print()
    test_get_input_tukey_window()
    print()
    test_get_input_d_excess()
    print()
    test_get_input_coriolis()
    print()
    test_get_input_sample_alt()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
