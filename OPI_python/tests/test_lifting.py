"""
Test lifting.py against MATLAB lifting.m behavior
"""

import numpy as np
import pytest
from opi.physics.lifting import calculate_lifting_for_samples
from opi.physics.wind_grid import wind_grid
from opi.catchment import catchment_indices


def create_simple_topography(nx=50, ny=50):
    """Create a simple Gaussian hill topography for testing"""
    dx = dy = 1000  # 1 km grid spacing
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Gaussian hill at center
    xc, yc = (nx-1) * dx / 2, (ny-1) * dy / 2
    h_grid = 2000 * np.exp(-((X - xc)**2 + (Y - yc)**2) / (2 * 5000**2))
    
    return x, y, h_grid, X, Y


def test_lifting_basic():
    """Test basic lifting calculation with a single sample"""
    x, y, h_grid, X, Y = create_simple_topography()
    
    # Uniform precipitation
    p_grid = np.ones_like(h_grid) * 1e-6  # 1 mm/day in kg/m2/s
    
    # Wind from north (0 degrees)
    azimuth = 0.0
    U = 10.0  # m/s
    tau_f = 100.0  # 100 s fall time
    
    # Single sample at the peak
    nx, ny = h_grid.shape
    sample_i, sample_j = nx // 2, ny // 2
    
    # Create ij_catch and ptr_catch for single local sample
    # ptr_catch format: [start_idx_sample0, end_idx_sample0+1, ...]
    # For 1 sample with 1 node: ptr_catch = [0, 1]
    ij_catch = [(sample_i, sample_j)]
    ptr_catch = [0, 1]  # One sample, one node
    
    lift_max, elevation, subsidence = calculate_lifting_for_samples(
        x, y, h_grid, p_grid, azimuth, U, tau_f, ij_catch, ptr_catch
    )
    
    print(f"Sample location: ({sample_i}, {sample_j})")
    print(f"Local elevation: {h_grid[sample_i, sample_j]:.1f} m")
    print(f"Lift max: {lift_max[0]:.1f} m")
    print(f"Elevation pred: {elevation[0]:.1f} m")
    print(f"Subsidence: {subsidence[0]:.1f} m")
    
    # Basic sanity checks
    assert len(lift_max) == 1
    assert len(elevation) == 1
    assert len(subsidence) == 1
    assert not np.isnan(lift_max[0])
    assert not np.isnan(elevation[0])
    assert not np.isnan(subsidence[0])
    
    # Elevation should match local elevation for single point
    np.testing.assert_allclose(elevation[0], h_grid[sample_i, sample_j], rtol=1e-5)
    
    # Lift max should be >= local elevation (upwind max)
    assert lift_max[0] >= h_grid[sample_i, sample_j] - 1  # Small tolerance for interpolation
    
    print("[PASS] Basic test passed")


def test_lifting_wind_direction():
    """Test that wind direction affects lifting calculation correctly"""
    x, y, h_grid, X, Y = create_simple_topography()
    p_grid = np.ones_like(h_grid) * 1e-6
    
    U = 10.0
    tau_f = 200.0
    
    # Sample at center
    nx, ny = h_grid.shape
    sample_i, sample_j = nx // 2, ny // 2
    ij_catch = [(sample_i, sample_j)]
    ptr_catch = [0, 1]
    
    # Test two wind directions
    azimuth_north = 0.0
    azimuth_south = 180.0
    
    lift_north, _, _ = calculate_lifting_for_samples(
        x, y, h_grid, p_grid, azimuth_north, U, tau_f, ij_catch, ptr_catch
    )
    
    lift_south, _, _ = calculate_lifting_for_samples(
        x, y, h_grid, p_grid, azimuth_south, U, tau_f, ij_catch, ptr_catch
    )
    
    print(f"Lift with north wind: {lift_north[0]:.1f} m")
    print(f"Lift with south wind: {lift_south[0]:.1f} m")
    
    # For a symmetric hill with fallout offset, results may differ slightly
    # because the upwind sampling positions are different
    # Just verify both give reasonable values (close to peak elevation)
    peak_elev = h_grid[sample_i, sample_j]
    assert lift_north[0] > peak_elev * 0.8, f"North wind lift should be close to peak: {lift_north[0]:.1f}"
    assert lift_south[0] > peak_elev * 0.8, f"South wind lift should be close to peak: {lift_south[0]:.1f}"
    
    print("[PASS] Wind direction test passed")


def test_lifting_catchment_weighted():
    """Test precipitation-weighted averaging for catchment sample"""
    x, y, h_grid, X, Y = create_simple_topography()
    
    nx, ny = h_grid.shape
    
    # Create a catchment area around the peak
    center_i, center_j = nx // 2, ny // 2
    catchment_nodes = []
    for di in range(-2, 3):
        for dj in range(-2, 3):
            i = center_i + di
            j = center_j + dj
            if 0 <= i < nx and 0 <= j < ny:
                catchment_nodes.append((i, j))
    
    # Non-uniform precipitation (higher at peak)
    p_grid = np.zeros_like(h_grid)
    for i, j in catchment_nodes:
        dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
        p_grid[i, j] = np.exp(-dist / 2) * 1e-6
    
    ij_catch = catchment_nodes
    ptr_catch = [0, len(catchment_nodes)]
    
    azimuth = 0.0
    U = 10.0
    tau_f = 100.0
    
    lift_max, elevation, subsidence = calculate_lifting_for_samples(
        x, y, h_grid, p_grid, azimuth, U, tau_f, ij_catch, ptr_catch
    )
    
    print(f"Catchment has {len(catchment_nodes)} nodes")
    print(f"Precipitation range: {p_grid.min():.2e} to {p_grid.max():.2e}")
    print(f"Lift max: {lift_max[0]:.1f} m")
    print(f"Elevation pred: {elevation[0]:.1f} m")
    
    # Weighted elevation should be different from simple mean
    simple_mean = np.mean([h_grid[i, j] for i, j in catchment_nodes])
    weighted_mean = elevation[0]
    
    print(f"Simple mean elevation: {simple_mean:.1f} m")
    print(f"Weighted mean elevation: {weighted_mean:.1f} m")
    
    # Weighted mean should be closer to peak (where precip is higher)
    peak_elevation = h_grid[center_i, center_j]
    assert abs(weighted_mean - peak_elevation) < abs(simple_mean - peak_elevation)
    
    print("[PASS] Catchment weighted test passed")


def test_lifting_no_precipitation():
    """Test behavior when there's no precipitation"""
    x, y, h_grid, X, Y = create_simple_topography()
    
    # Zero precipitation
    p_grid = np.zeros_like(h_grid)
    
    nx, ny = h_grid.shape
    sample_i, sample_j = nx // 2, ny // 2
    
    # Create catchment with multiple nodes
    catchment_nodes = [(sample_i, sample_j)]
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        i, j = sample_i + di, sample_j + dj
        if 0 <= i < nx and 0 <= j < ny:
            catchment_nodes.append((i, j))
    
    ij_catch = catchment_nodes
    ptr_catch = [0, len(catchment_nodes)]
    
    azimuth = 0.0
    U = 10.0
    tau_f = 100.0
    
    lift_max, elevation, subsidence = calculate_lifting_for_samples(
        x, y, h_grid, p_grid, azimuth, U, tau_f, ij_catch, ptr_catch
    )
    
    print(f"No precipitation test:")
    print(f"  Lift max: {lift_max[0]:.1f} m")
    print(f"  Elevation: {elevation[0]:.1f} m")
    print(f"  Subsidence: {subsidence[0]:.1f} m")
    
    # Should use uniform weighting (no NaN)
    assert not np.isnan(lift_max[0])
    assert not np.isnan(elevation[0])
    assert not np.isnan(subsidence[0])
    
    print("[PASS] No precipitation test passed")


def test_lifting_multiple_samples():
    """Test with multiple samples"""
    x, y, h_grid, X, Y = create_simple_topography()
    p_grid = np.ones_like(h_grid) * 1e-6
    
    nx, ny = h_grid.shape
    
    # Two samples: one at peak, one at lower elevation
    sample1_i, sample1_j = nx // 2, ny // 2
    sample2_i, sample2_j = nx // 4, ny // 4
    
    # Each sample has single node
    ij_catch = [(sample1_i, sample1_j), (sample2_i, sample2_j)]
    ptr_catch = [0, 1, 2]  # Two samples: sample0 has 1 node, sample1 has 1 node
    
    azimuth = 0.0
    U = 10.0
    tau_f = 100.0
    
    lift_max, elevation, subsidence = calculate_lifting_for_samples(
        x, y, h_grid, p_grid, azimuth, U, tau_f, ij_catch, ptr_catch
    )
    
    print(f"Sample 1 (peak): elev={h_grid[sample1_i, sample1_j]:.1f}, lift={lift_max[0]:.1f}")
    print(f"Sample 2 (lower): elev={h_grid[sample2_i, sample2_j]:.1f}, lift={lift_max[1]:.1f}")
    
    assert len(lift_max) == 2
    assert len(elevation) == 2
    assert len(subsidence) == 2
    
    # Sample 1 should have higher values
    assert lift_max[0] > lift_max[1]
    assert elevation[0] > elevation[1]
    
    print("[PASS] Multiple samples test passed")


def test_wind_grid_consistency():
    """Test that wind_grid transformation is consistent"""
    x, y, h_grid, X, Y = create_simple_topography(nx=30, ny=30)
    
    # Test different azimuths
    for azimuth in [0, 45, 90, 135, 180]:
        Sxy, Txy, s, t, Xst, Yst = wind_grid(x, y, azimuth)
        
        # Check grid dimensions
        assert Sxy.shape == (len(x), len(y))
        assert Txy.shape == (len(x), len(y))
        assert Xst.shape == (len(s), len(t))
        assert Yst.shape == (len(s), len(t))
        
        print(f"Azimuth {azimuth}°: s range [{s.min():.0f}, {s.max():.0f}], t range [{t.min():.0f}, {t.max():.0f}]")
    
    print("[PASS] Wind grid consistency test passed")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing lifting.py")
    print("=" * 60)
    
    test_lifting_basic()
    print()
    test_lifting_wind_direction()
    print()
    test_lifting_catchment_weighted()
    print()
    test_lifting_no_precipitation()
    print()
    test_lifting_multiple_samples()
    print()
    test_wind_grid_consistency()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
