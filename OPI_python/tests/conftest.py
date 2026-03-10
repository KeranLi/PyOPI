"""
Pytest configuration and shared fixtures for OPI tests.
"""

import numpy as np
import pytest


@pytest.fixture
def standard_grid():
    """Standard 100x100 km grid with 1 km resolution."""
    dx = 1000.0
    nx, ny = 101, 101
    x = np.arange(nx) * dx - (nx // 2) * dx
    y = np.arange(ny) * dx - (ny // 2) * dx
    return {'x': x, 'y': y, 'dx': dx, 'nx': nx, 'ny': ny}


@pytest.fixture
def gaussian_mountain(standard_grid):
    """Gaussian mountain topography (2000 m peak, 20 km width)."""
    x, y = standard_grid['x'], standard_grid['y']
    X, Y = np.meshgrid(x, y)
    h_grid = 2000.0 * np.exp(-(X**2 + Y**2) / (2 * 20000**2))
    return {
        'x': x,
        'y': y,
        'h_grid': h_grid,
        'X': X,
        'Y': Y,
        'height': 2000.0,
        'width': 20000.0
    }


@pytest.fixture
def standard_parameters():
    """Standard OPI parameters for testing."""
    return {
        'U': 10.0,
        'azimuth': 270.0,
        'T0': 288.0,
        'M': 0.8,
        'kappa': 0.0,
        'tau_c': 1000.0,
        'd2H0': -0.100,
        'd_d2H0_d_lat': 0.0,
        'f_p0': 1.0
    }


@pytest.fixture
def model_constants():
    """Standard model constants."""
    return {
        'f_c': 1e-4,
        'h_r': 540.0,
        'sd_res_ratio': 28.3
    }
