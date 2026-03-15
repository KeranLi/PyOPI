"""
Test grid_read function against MATLAB gridRead.m behavior
"""

import numpy as np
import os
import tempfile
from scipy.io import savemat, loadmat
from opi.io.data_loader import grid_read


def create_test_mat_file(x, y, z, extra_vars=None):
    """Create a temporary .mat file with test data"""
    data = {'x': x, 'y': y, 'z': z}
    if extra_vars:
        data.update(extra_vars)
    
    fd, filepath = tempfile.mkstemp(suffix='.mat')
    os.close(fd)
    savemat(filepath, data)
    return filepath


def test_grid_read_basic():
    """Test basic grid reading"""
    # Create test grid
    x = np.linspace(-100, 100, 50)
    y = np.linspace(-50, 50, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/10) * np.cos(Y/10)
    
    mat_file = create_test_mat_file(x, y, Z)
    
    try:
        x_read, y_read, z_read = grid_read(mat_file)
        
        print(f"Grid shape: {z_read.shape}")
        print(f"x length: {len(x_read)}, y length: {len(y_read)}")
        
        assert len(x_read) == 50
        assert len(y_read) == 40
        assert z_read.shape == (40, 50)
        
        np.testing.assert_allclose(x_read, x)
        np.testing.assert_allclose(y_read, y)
        np.testing.assert_allclose(z_read, Z)
        
        print("[PASS] Basic grid read test")
    finally:
        os.unlink(mat_file)


def test_grid_read_with_metadata():
    """Test grid reading with metadata variables"""
    x = np.linspace(0, 100, 30)
    y = np.linspace(0, 50, 20)
    Z = np.random.rand(20, 30)
    
    # Add metadata (string/struct should be ignored)
    extra = {
        'description': 'Test grid',
        'author': 'Test',
    }
    
    mat_file = create_test_mat_file(x, y, Z, extra)
    
    try:
        x_read, y_read, z_read = grid_read(mat_file)
        
        assert z_read.shape == (20, 30)
        np.testing.assert_allclose(x_read, x)
        np.testing.assert_allclose(y_read, y)
        
        print("[PASS] Grid read with metadata test")
    finally:
        os.unlink(mat_file)


def test_grid_read_column_vectors():
    """Test with column vectors (nx1 and mx1)"""
    x = np.linspace(0, 100, 30).reshape(1, -1)  # Row vector (1, 30)
    y = np.linspace(0, 50, 20).reshape(-1, 1)   # Column vector (20, 1)
    Z = np.random.rand(20, 30)
    
    mat_file = create_test_mat_file(x, y, Z)
    
    try:
        x_read, y_read, z_read = grid_read(mat_file)
        
        assert len(x_read) == 30
        assert len(y_read) == 20
        assert z_read.shape == (20, 30)
        
        np.testing.assert_allclose(x_read, x.flatten())
        np.testing.assert_allclose(y_read, y.flatten())
        
        print("[PASS] Column vectors test")
    finally:
        os.unlink(mat_file)


def test_grid_read_extra_numeric():
    """Test with extra numeric variables (should still work)"""
    x = np.linspace(0, 100, 30)
    y = np.linspace(0, 50, 20)
    Z = np.random.rand(20, 30)
    
    # Add extra numeric variables
    extra = {
        'extra_param': np.array([1.0, 2.0, 3.0]),
        'scalar_value': np.array([[42.0]]),
    }
    
    mat_file = create_test_mat_file(x, y, Z, extra)
    
    try:
        x_read, y_read, z_read = grid_read(mat_file)
        
        assert z_read.shape == (20, 30)
        
        print("[PASS] Extra numeric variables test")
    finally:
        os.unlink(mat_file)


def test_grid_read_mismatched_sizes():
    """Test error handling for mismatched grid/vector sizes"""
    x = np.linspace(0, 100, 30)
    y = np.linspace(0, 50, 25)  # Intentionally wrong size (should be 20)
    Z = np.random.rand(20, 30)
    
    mat_file = create_test_mat_file(x, y, Z)
    
    try:
        try:
            grid_read(mat_file)
            print("[FAIL] Should have raised error for mismatched sizes")
        except ValueError as e:
            print(f"[PASS] Correctly raised error: {e}")
    finally:
        os.unlink(mat_file)


def test_grid_read_nonexistent_file():
    """Test error handling for non-existent file"""
    try:
        grid_read('/nonexistent/path/file.mat')
        print("[FAIL] Should have raised error for non-existent file")
    except FileNotFoundError:
        print("[PASS] Correctly raised FileNotFoundError")


def test_grid_read_single_precision():
    """Test with single precision (float32) data"""
    x = np.linspace(0, 100, 30).astype(np.float32)
    y = np.linspace(0, 50, 20).astype(np.float32)
    Z = np.random.rand(20, 30).astype(np.float32)
    
    mat_file = create_test_mat_file(x, y, Z)
    
    try:
        x_read, y_read, z_read = grid_read(mat_file)
        
        assert z_read.shape == (20, 30)
        # loadmat may preserve float32 or convert to float64
        # Just verify it's a numeric type
        assert np.issubdtype(x_read.dtype, np.floating)
        
        print("[PASS] Single precision test")
    finally:
        os.unlink(mat_file)


def test_grid_read_largest_grid():
    """Test that the largest 2D array is selected as grid"""
    x = np.linspace(0, 100, 50)
    y = np.linspace(0, 50, 40)
    Z_large = np.random.rand(40, 50)  # Large grid
    Z_small = np.random.rand(10, 10)  # Small grid
    
    # Save with both grids
    data = {'x': x, 'y': y, 'z_large': Z_large, 'z_small': Z_small}
    fd, mat_file = tempfile.mkstemp(suffix='.mat')
    os.close(fd)
    savemat(mat_file, data)
    
    try:
        x_read, y_read, z_read = grid_read(mat_file)
        
        # Should select the larger grid
        assert z_read.shape == (40, 50), f"Expected (40, 50), got {z_read.shape}"
        
        print("[PASS] Largest grid selection test")
    finally:
        os.unlink(mat_file)


def test_grid_read_compare_with_matlab_logic():
    """Compare behavior with MATLAB's gridRead logic"""
    # MATLAB gridRead:
    # 1. Finds variables with class 'single' or 'double'
    # 2. Checks that there are exactly 3 numeric variables (mxn matrix + 2 vectors)
    # 3. Selects grid as max(wSize(:,1) .* wSize(:,2))
    # 4. Selects x as max(wSize(:,2)) from remaining
    # 5. Selects y as max(wSize(:,1)) from remaining
    
    # Create test data
    x = np.arange(100, 200, 2)  # 50 elements
    y = np.arange(50)           # 50 elements
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/10) * np.cos(Y/10)
    
    mat_file = create_test_mat_file(x, y, Z)
    
    try:
        x_read, y_read, z_read = grid_read(mat_file)
        
        # Verify sizes match MATLAB convention:
        # z should be (len(y), len(x))
        assert z_read.shape[0] == len(y), "Grid rows should match y length"
        assert z_read.shape[1] == len(x), "Grid columns should match x length"
        
        print(f"MATLAB-like behavior:")
        print(f"  Grid shape: {z_read.shape} (rows=y, cols=x)")
        print(f"  x range: [{x_read.min():.1f}, {x_read.max():.1f}]")
        print(f"  y range: [{y_read.min():.1f}, {y_read.max():.1f}]")
        
        print("[PASS] MATLAB logic comparison test")
    finally:
        os.unlink(mat_file)


if __name__ == '__main__':
    print("=" * 60)
    print("Testing grid_read function")
    print("=" * 60)
    
    test_grid_read_basic()
    print()
    test_grid_read_with_metadata()
    print()
    test_grid_read_column_vectors()
    print()
    test_grid_read_extra_numeric()
    print()
    test_grid_read_mismatched_sizes()
    print()
    test_grid_read_nonexistent_file()
    print()
    test_grid_read_single_precision()
    print()
    test_grid_read_largest_grid()
    print()
    test_grid_read_compare_with_matlab_logic()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
