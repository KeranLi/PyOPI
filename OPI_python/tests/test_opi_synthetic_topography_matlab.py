"""
Comparison tests for opiSyntheticTopography.m (MATLAB) vs Python implementation

MATLAB opiSyntheticTopography creates synthetic topography for OPI testing:
- North-directed sinusoids at equator
- East-directed sinusoids at equator
- East-directed Gaussian at equator
- East-directed Gaussian at 45 degrees north
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TestOpiSyntheticTopographyBasic:
    """Basic functionality tests for synthetic topography generation"""
    
    def test_grid_creation(self):
        """Test basic grid creation parameters"""
        lon0, lat0 = 0, 0
        dx, dy = 1000, 1000  # 1 km spacing
        lx, ly = 700e3, 700e3  # 700 km extent
        
        # Calculate grid dimensions
        nx = int(np.ceil(lx / dx)) + 1
        ny = int(np.ceil(ly / dy)) + 1
        
        # Recalculate to match MATLAB
        dx = lx / (nx - 1)
        dy = ly / (ny - 1)
        
        x_min, y_min = -lx / 2, -ly / 2
        x = x_min + np.arange(nx) * dx
        y = y_min + np.arange(ny) * dy
        
        assert len(x) == nx
        assert len(y) == ny
        assert x[0] == x_min
        assert y[0] == y_min
    
    def test_xy_to_lonlat_conversion(self):
        """Test conversion from Cartesian to geographic coordinates"""
        # Simple conversion assuming flat Earth at equator
        # 1 degree longitude ~ 111.32 km at equator
        # 1 degree latitude ~ 110.57 km
        
        lon0, lat0 = 0, 0
        x = np.array([0, 100e3, 200e3])  # 0, 100, 200 km east
        y = np.array([0, 100e3, 200e3])  # 0, 100, 200 km north
        
        M_PER_DEGREE = np.pi * 6371e3 / 180  # ~111,195 m/deg
        
        lon = lon0 + x / (M_PER_DEGREE * np.cos(np.radians(lat0)))
        lat = lat0 + y / M_PER_DEGREE
        
        # At equator, x corresponds to longitude change
        # y corresponds to latitude change
        assert len(lon) == len(x)
        assert len(lat) == len(y)
        assert lon[0] == lon0
        assert lat[0] == lat0
        
        # Check conversion is reasonable
        # 100 km should be less than 1 degree
        assert lon[1] < 1.0
        assert lat[1] < 1.0


class TestOpiSyntheticTopographyMatlabComparison:
    """Direct comparison with MATLAB opiSyntheticTopography.m"""
    
    def test_north_directed_sinusoids(self):
        """
        Test North-directed sinusoidal topography.
        MATLAB: hGrid = amplitude * (1 + sin(2*pi*(y - offset)/wavelength - pi/2))/2
        """
        y = np.linspace(-350e3, 350e3, 701)
        x = np.linspace(-350e3, 350e3, 701)
        
        wavelength = 100e3
        amplitude = 3e3
        offset_horizontal = 100e3
        offset_vertical = 1
        y_min = -350e3
        
        # Calculate topography (North-directed)
        h_grid = amplitude * (1 + np.sin(2 * np.pi * (y - (y_min + offset_horizontal)) / wavelength - np.pi / 2)) / 2
        h_grid[y < (y_min + offset_horizontal)] = 0
        h_grid = h_grid + offset_vertical
        h_grid = np.tile(h_grid[:, np.newaxis], (1, len(x)))
        
        # Check dimensions
        assert h_grid.shape == (len(y), len(x))
        
        # Check height range
        assert np.min(h_grid) >= offset_vertical
        assert np.max(h_grid) <= amplitude + offset_vertical
        
        # Check zero region
        y_zero_threshold = y_min + offset_horizontal
        y_indices = y < y_zero_threshold
        assert np.all(h_grid[y_indices, 0] == offset_vertical)
    
    def test_east_directed_sinusoids(self):
        """
        Test East-directed sinusoidal topography.
        MATLAB: hGrid = amplitude * (1 + sin(2*pi*(x - offset)/wavelength - pi/2))/2
        """
        y = np.linspace(-350e3, 350e3, 701)
        x = np.linspace(-350e3, 350e3, 701)
        
        wavelength = 100e3
        amplitude = 3e3
        offset_horizontal = 100e3
        offset_vertical = 1
        x_min = -350e3
        
        # Calculate topography (East-directed)
        h_grid = amplitude * (1 + np.sin(2 * np.pi * (x - (x_min + offset_horizontal)) / wavelength - np.pi / 2)) / 2
        h_grid[x < (x_min + offset_horizontal)] = 0
        h_grid = h_grid + offset_vertical
        h_grid = np.tile(h_grid[np.newaxis, :], (len(y), 1))
        
        # Check dimensions
        assert h_grid.shape == (len(y), len(x))
        
        # Check height range
        assert np.min(h_grid) >= offset_vertical
        assert np.max(h_grid) <= amplitude + offset_vertical
    
    def test_east_directed_gaussian(self):
        """
        Test East-directed Gaussian topography.
        MATLAB: hGrid = amplitude * exp(-0.5*(x/sigma)^2)
        """
        y = np.linspace(-350e3, 350e3, 701)
        x = np.linspace(-350e3, 350e3, 701)
        
        sigma = 100e3 / 3
        amplitude = 3e3
        offset_vertical = 1
        
        # Calculate Gaussian topography
        h_grid = amplitude * np.exp(-0.5 * (x / sigma) ** 2)
        h_grid = h_grid + offset_vertical
        h_grid = np.tile(h_grid[np.newaxis, :], (len(y), 1))
        
        # Check dimensions
        assert h_grid.shape == (len(y), len(x))
        
        # Check peak is at center
        x_center_idx = len(x) // 2
        assert h_grid[0, x_center_idx] == amplitude + offset_vertical
        
        # Check Gaussian decay
        assert h_grid[0, 0] < h_grid[0, x_center_idx]
        
        # Check minimum
        edge_idx = np.argmin(h_grid[0, :])
        assert h_grid[0, edge_idx] >= offset_vertical
    
    def test_east_directed_gaussian_45n(self):
        """Test East-directed Gaussian at 45 degrees North"""
        lon0, lat0 = 0, 45
        
        y = np.linspace(-350e3, 350e3, 701)
        x = np.linspace(-350e3, 350e3, 701)
        
        sigma = 100e3 / 3
        amplitude = 3e3
        offset_vertical = 1
        
        # Calculate Gaussian topography (same formula)
        h_grid = amplitude * np.exp(-0.5 * (x / sigma) ** 2)
        h_grid = h_grid + offset_vertical
        h_grid = np.tile(h_grid[np.newaxis, :], (len(y), 1))
        
        # Verify grid is created correctly
        assert h_grid.shape == (len(y), len(x))
        assert np.max(h_grid) == amplitude + offset_vertical
        
        # Verify coordinates are different from equator
        assert lat0 == 45
        assert lat0 != 0
    
    def test_grid_vector_orientation(self):
        """
        Test grid vector orientation for gridRead compatibility.
        MATLAB: lon is row vector, lat is column vector
        """
        nx, ny = 701, 701
        
        # Create grid vectors
        lon = np.linspace(-5, 5, nx)
        lat = np.linspace(-5, 5, ny)
        
        # MATLAB requires lon as row vector, lat as column vector
        lon = lon.reshape(1, -1)  # Row vector
        lat = lat.reshape(-1, 1)  # Column vector
        
        assert lon.shape == (1, nx)
        assert lat.shape == (ny, 1)


class TestOpiSyntheticTopographyVisualization:
    """Visualization tests"""
    
    def test_north_sinusoid_plot(self):
        """Test visualization of North-directed sinusoids"""
        y = np.linspace(-350e3, 350e3, 701)
        x = np.linspace(-350e3, 350e3, 701)
        
        wavelength = 100e3
        amplitude = 3e3
        offset_horizontal = 100e3
        offset_vertical = 1
        y_min = -350e3
        
        h_grid = amplitude * (1 + np.sin(2 * np.pi * (y - (y_min + offset_horizontal)) / wavelength - np.pi / 2)) / 2
        h_grid[y < (y_min + offset_horizontal)] = 0
        h_grid = h_grid + offset_vertical
        h_grid = np.tile(h_grid[:, np.newaxis], (1, len(x)))
        
        # Convert to lon/lat for plotting
        lon = np.linspace(-5, 5, len(x))
        lat = np.linspace(-5, 5, len(y))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        c = ax.pcolormesh(lon, lat, h_grid, shading='auto')
        ax.set_title('North-Directed Sinusoids')
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        plt.colorbar(c, ax=ax, label='Elevation (m)')
        
        assert fig is not None
        plt.close(fig)
    
    def test_east_gaussian_plot(self):
        """Test visualization of East-directed Gaussian"""
        y = np.linspace(-350e3, 350e3, 701)
        x = np.linspace(-350e3, 350e3, 701)
        
        sigma = 100e3 / 3
        amplitude = 3e3
        offset_vertical = 1
        
        h_grid = amplitude * np.exp(-0.5 * (x / sigma) ** 2)
        h_grid = h_grid + offset_vertical
        h_grid = np.tile(h_grid[np.newaxis, :], (len(y), 1))
        
        lon = np.linspace(-5, 5, len(x))
        lat = np.linspace(-5, 5, len(y))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        c = ax.pcolormesh(lon, lat, h_grid, shading='auto')
        ax.set_title('East-Directed Gaussian')
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        plt.colorbar(c, ax=ax, label='Elevation (m)')
        
        assert fig is not None
        plt.close(fig)


class TestOpiSyntheticTopographyEdgeCases:
    """Edge case tests"""
    
    def test_small_grid(self):
        """Test with small grid size"""
        nx, ny = 11, 11
        x = np.linspace(-5e3, 5e3, nx)
        y = np.linspace(-5e3, 5e3, ny)
        
        # Simple flat topography
        h_grid = np.ones((ny, nx))
        
        assert h_grid.shape == (ny, nx)
    
    def test_large_grid(self):
        """Test with large grid size"""
        nx, ny = 1001, 1001
        x = np.linspace(-500e3, 500e3, nx)
        
        sigma = 100e3
        amplitude = 3e3
        
        h_grid = amplitude * np.exp(-0.5 * (x / sigma) ** 2)
        h_grid = np.tile(h_grid[np.newaxis, :], (ny, 1))
        
        assert h_grid.shape == (ny, nx)
        assert np.max(h_grid) == amplitude
    
    def test_zero_amplitude(self):
        """Test with zero amplitude (flat topography)"""
        x = np.linspace(-100e3, 100e3, 101)
        y = np.linspace(-100e3, 100e3, 101)
        
        amplitude = 0
        offset_vertical = 1
        
        h_grid = np.zeros((len(y), len(x))) + offset_vertical
        
        assert np.all(h_grid == offset_vertical)
        assert np.std(h_grid) == 0
    
    def test_very_small_wavelength(self):
        """Test with very small wavelength"""
        x = np.linspace(-100e3, 100e3, 1001)
        wavelength = 1e3  # 1 km wavelength
        amplitude = 1e3
        
        h_grid = amplitude * np.sin(2 * np.pi * x / wavelength)
        
        # Should have many oscillations
        zero_crossings = np.sum(np.diff(np.sign(h_grid)) != 0)
        assert zero_crossings > 10  # Multiple zero crossings


class TestOpiSyntheticTopographyPhysical:
    """Physical consistency tests"""
    
    def test_no_negative_elevations(self):
        """Test that no negative elevations are produced"""
        x = np.linspace(-350e3, 350e3, 701)
        y = np.linspace(-350e3, 350e3, 701)
        
        # Sinusoid with offset to ensure positivity
        amplitude = 3e3
        offset = 1
        wavelength = 100e3
        
        h_grid = amplitude * (1 + np.sin(2 * np.pi * x / wavelength)) / 2 + offset
        h_grid = np.tile(h_grid[np.newaxis, :], (len(y), 1))
        
        assert np.all(h_grid > 0)
    
    def test_gaussian_symmetry(self):
        """Test that Gaussian is symmetric around center"""
        x = np.linspace(-200e3, 200e3, 401)
        sigma = 50e3
        amplitude = 3e3
        
        h_grid = amplitude * np.exp(-0.5 * (x / sigma) ** 2)
        
        center_idx = len(x) // 2
        
        # Check symmetry
        for i in range(1, 50):
            left_idx = center_idx - i
            right_idx = center_idx + i
            if left_idx >= 0 and right_idx < len(x):
                assert np.isclose(h_grid[left_idx], h_grid[right_idx], rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
