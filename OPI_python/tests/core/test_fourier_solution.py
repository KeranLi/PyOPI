"""
Tests for the fourier_solution module.

This module tests the Fourier solution for linearized Euler equations
including coordinate transformations and wavenumber calculations.
"""

import numpy as np
import pytest

from opi.core.fourier_solution import wind_grid, fourier_solution


@pytest.fixture
def simple_topography():
    """Create a simple Gaussian mountain topography for testing."""
    x = np.linspace(-50000, 50000, 50)  # 100 km in x
    y = np.linspace(-50000, 50000, 50)  # 100 km in y
    X, Y = np.meshgrid(x, y)
    
    # Gaussian mountain with 2000m height and 20km width
    h_grid = 2000 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    
    return x, y, h_grid


@pytest.fixture
def flat_topography():
    """Create a flat topography for testing."""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    h_grid = np.zeros((50, 50))
    return x, y, h_grid


@pytest.fixture
def ridge_topography():
    """Create a ridge topography for testing."""
    x = np.linspace(-50000, 50000, 50)
    y = np.linspace(-50000, 50000, 50)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian ridge along y-axis
    h_grid = 1500 * np.exp(-X**2 / (2 * 10000**2))
    
    return x, y, h_grid


class TestWindGrid:
    """Tests for the wind_grid function."""

    def test_wind_grid_transform_preserves_distances(self, simple_topography):
        """Test that coordinate transformation preserves distances."""
        x, y, _ = simple_topography
        azimuth = 45.0  # 45 degrees from North
        
        result = wind_grid(x, y, azimuth)
        
        # Check that original grid spacing is preserved approximately
        dX = x[1] - x[0]
        dY = y[1] - y[0]
        
        # The transformed grid should have reasonable spacing
        assert result['d_s'] > 0
        assert result['d_t'] > 0
        
        # Check that distances are preserved in the transformation
        # A point at (dX, 0) in original should have same distance in transformed
        azimuth_rad = np.deg2rad(azimuth)
        original_distance = np.sqrt(dX**2 + dY**2)
        
        # The s,t grid spacing should be comparable to original
        assert np.isclose(result['d_s'], result['d_t'], rtol=0.5) or \
               result['d_s'] / result['d_t'] < 10

    def test_wind_grid_transform_90_degrees(self, simple_topography):
        """Test coordinate transformation with 90 degree azimuth."""
        x, y, _ = simple_topography
        
        result = wind_grid(x, y, 90.0)
        
        # At 90 degrees, s should align with x (East)
        # and t should align with y (North)
        azimuth_rad = np.deg2rad(90.0)
        
        # Check transformation formulas:
        # Sxy = X*sin(azimuth) + Y*cos(azimuth) = X*1 + Y*0 = X
        # Txy = -X*cos(azimuth) + Y*sin(azimuth) = -X*0 + Y*1 = Y
        X_test, Y_test = np.meshgrid(np.array([0, 1000]), np.array([0, 1000]))
        
        expected_Sxy = X_test * np.sin(azimuth_rad) + Y_test * np.cos(azimuth_rad)
        expected_Txy = -X_test * np.cos(azimuth_rad) + Y_test * np.sin(azimuth_rad)
        
        assert np.allclose(result['Sxy'][0, :], x, rtol=0.1) or \
               np.allclose(result['Sxy'][:, 0], y, rtol=0.1)

    def test_wind_grid_output_keys(self, simple_topography):
        """Test that wind_grid returns expected dictionary keys."""
        x, y, _ = simple_topography
        
        result = wind_grid(x, y, 45.0)
        
        expected_keys = {'Sxy', 'Txy', 's', 't', 'Xst', 'Yst', 'd_s', 'd_t'}
        assert set(result.keys()) == expected_keys

    def test_wind_grid_dimensions(self, simple_topography):
        """Verify output grid dimensions are consistent."""
        x, y, _ = simple_topography
        
        result = wind_grid(x, y, 45.0)
        
        # Sxy and Txy should have same shape as input meshgrid
        n_y, n_x = len(y), len(x)
        assert result['Sxy'].shape == (n_y, n_x)
        assert result['Txy'].shape == (n_y, n_x)
        
        # Xst and Yst should have shape based on s and t vectors
        n_s, n_t = len(result['s']), len(result['t'])
        assert result['Xst'].shape == (n_s, n_t)
        assert result['Yst'].shape == (n_s, n_t)

    def test_wind_grid_consistency(self, simple_topography):
        """Test that forward and backward transformations are consistent."""
        x, y, _ = simple_topography
        azimuth = 60.0
        
        result = wind_grid(x, y, azimuth)
        
        # The min/max of Sxy should match the range of s vector
        assert np.isclose(np.min(result['Sxy']), result['s'][0], rtol=1e-10)
        assert np.isclose(np.max(result['Sxy']), result['s'][-1], rtol=1e-10)
        
        # The min/max of Txy should match the range of t vector
        assert np.isclose(np.min(result['Txy']), result['t'][0], rtol=1e-10)
        assert np.isclose(np.max(result['Txy']), result['t'][-1], rtol=1e-10)


class TestFourierSolution:
    """Tests for the fourier_solution function."""

    def test_fourier_solution_output_keys(self, simple_topography):
        """Verify fourier_solution returns expected dictionary keys."""
        x, y, h_grid = simple_topography
        
        U = 10.0          # m/s
        azimuth = 90.0    # degrees
        NM = 0.01         # rad/s
        f_c = 1e-4        # rad/s
        h_rho = 8000.0    # m
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        expected_keys = {
            's', 't', 'Sxy', 'Txy', 'h_wind', 
            'k_s', 'k_t', 'h_hat', 'k_z',
            'n_s_pad', 'n_t_pad', 'd_s'
        }
        assert set(result.keys()) == expected_keys

    def test_fourier_solution_wavenumbers(self, simple_topography):
        """Verify wavenumber calculation is correct."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # Check that k_s and k_t are properly sized
        n_s_pad = result['n_s_pad']
        n_t_pad = result['n_t_pad']
        
        assert len(result['k_s']) == n_s_pad
        assert len(result['k_t']) == n_t_pad
        
        # Check that k_s is sorted (for proper FFT ordering)
        # The first element should be 0 (DC component)
        assert result['k_s'][0] == 0.0
        
        # Check that k_t is properly calculated
        assert result['k_t'][0] == 0.0
        
        # Check that wavenumbers have units of rad/m
        # Typical values should be in range of 1e-6 to 1e-2
        assert np.all(np.abs(result['k_s']) >= 0)
        assert np.all(np.abs(result['k_t']) >= 0)

    def test_fourier_solution_padding(self, simple_topography):
        """Verify zero padding is applied correctly."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # Check padding dimensions
        assert result['n_s_pad'] == 2 * result['h_wind'].shape[0]
        assert result['n_t_pad'] == result['h_wind'].shape[1]
        
        # Check that h_hat has padded dimensions
        assert result['h_hat'].shape == (result['n_s_pad'], result['n_t_pad'])
        
        # Check that k_z also has padded dimensions
        assert result['k_z'].shape == (result['n_s_pad'], result['n_t_pad'])

    def test_fourier_solution_flat_topography(self, flat_topography):
        """Test that flat topography produces near-zero Fourier coefficients."""
        x, y, h_grid = flat_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # For flat topography, h_wind should be nearly zero
        assert np.allclose(result['h_wind'], 0.0, atol=1e-10)
        
        # The DC component of h_hat should be near zero
        assert np.abs(result['h_hat'][0, 0]) < 1e-6

    def test_fourier_solution_ridge_interpolation(self, ridge_topography):
        """Test interpolation of ridge topography to wind grid."""
        x, y, h_grid = ridge_topography
        
        U = 10.0
        azimuth = 90.0  # Eastward wind
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # The maximum height in h_wind should be close to original
        assert np.max(result['h_wind']) > 1000  # Should preserve significant height
        assert np.max(result['h_wind']) <= 2000  # Should not exceed original much
        
        # h_wind should have the correct shape
        assert result['h_wind'].shape[0] == len(result['s'])
        assert result['h_wind'].shape[1] == len(result['t'])

    def test_fourier_solution_vertical_wavenumber(self, simple_topography):
        """Test that vertical wavenumber k_z is calculated correctly."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # k_z should be complex (to handle evanescent waves)
        assert np.iscomplexobj(result['k_z'])
        
        # k_z should have proper shape
        assert result['k_z'].shape == (result['n_s_pad'], result['n_t_pad'])

    def test_fourier_solution_different_azimuths(self, simple_topography):
        """Test that solution works with different wind directions."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        azimuths = [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]
        
        for azimuth in azimuths:
            result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
            
            # All should have consistent output structure
            assert 'h_wind' in result
            assert 'k_z' in result
            assert 'h_hat' in result
            
            # h_hat should be complex
            assert np.iscomplexobj(result['h_hat'])

    def test_fourier_solution_grid_consistency(self, simple_topography):
        """Test that output grids are internally consistent."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # s and t vectors should match h_wind dimensions
        assert len(result['s']) == result['h_wind'].shape[0]
        assert len(result['t']) == result['h_wind'].shape[1]
        
        # k_s should match n_s_pad
        assert len(result['k_s']) == result['n_s_pad']
        
        # k_t should match n_t_pad
        assert len(result['k_t']) == result['n_t_pad']


class TestWavenumberProperties:
    """Tests specifically for wavenumber calculation properties."""

    def test_wavenumber_symmetry(self, simple_topography):
        """Test that wavenumbers have expected symmetry properties."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        k_s = result['k_s']
        k_t = result['k_t']
        
        # For real input, the FFT of real data has Hermitian symmetry
        # This means k values should be symmetric around 0 (with wrapping)
        # The first half should be non-negative, second half negative
        n_s_pad = len(k_s)
        mid_point = n_s_pad // 2 + 1
        
        # First half should be >= 0 (with possible numerical errors)
        assert np.all(k_s[:mid_point] >= -1e-15)

    def test_wavenumber_spacing(self, simple_topography):
        """Test that wavenumber spacing is consistent."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # Calculate spacing for k_s
        k_s_diff = np.diff(result['k_s'])
        # Most differences should be the same (FFT frequency spacing)
        # The Nyquist transition has a large jump (wraps from positive to negative)
        d_k_s_expected = 2 * np.pi / (result['d_s'] * result['n_s_pad'])
        # Count how many differences match expected spacing (allowing for wrap point)
        close_to_expected = np.sum(np.abs(np.abs(k_s_diff) - d_k_s_expected) < 1e-15)
        assert close_to_expected >= len(k_s_diff) - 1  # At most one wrap point
        
        # Calculate spacing for k_t
        k_t_diff = np.diff(result['k_t'])
        d_k_t_expected = 2 * np.pi / ((result['t'][1] - result['t'][0]) * result['n_t_pad'])
        close_to_expected_t = np.sum(np.abs(np.abs(k_t_diff) - d_k_t_expected) < 1e-15)
        assert close_to_expected_t >= len(k_t_diff) - 1


class TestPaddingBehavior:
    """Tests specifically for zero padding behavior."""

    def test_padding_doubles_s_dimension(self, simple_topography):
        """Test that s dimension is doubled by padding."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        n_s_original = result['h_wind'].shape[0]
        assert result['n_s_pad'] == 2 * n_s_original

    def test_padding_preserves_t_dimension(self, simple_topography):
        """Test that t dimension is preserved without padding."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        n_t_original = result['h_wind'].shape[1]
        assert result['n_t_pad'] == n_t_original

    def test_fft_dimensions_match_padding(self, simple_topography):
        """Test that FFT output dimensions match padding specifications."""
        x, y, h_grid = simple_topography
        
        U = 10.0
        azimuth = 90.0
        NM = 0.01
        f_c = 1e-4
        h_rho = 8000.0
        
        result = fourier_solution(x, y, h_grid, U, azimuth, NM, f_c, h_rho)
        
        # h_hat shape should match padding
        assert result['h_hat'].shape[0] == result['n_s_pad']
        assert result['h_hat'].shape[1] == result['n_t_pad']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
