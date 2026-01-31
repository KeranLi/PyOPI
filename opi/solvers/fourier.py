"""
Fourier solver for terrain transformation.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from ..core.base import OPIBase
from ..models.config import SolverConfig
from ..models.domain import Topography
from ..models.parameters import WindField, AtmosphereState
from ..models.results import FourierSolution


class FourierSolver(OPIBase):
    """
    FFT-based solver for linearized Euler equations.
    
    Transforms topography to wind-aligned coordinates and computes
    Fourier coefficients and vertical wavenumbers.
    """
    
    def __init__(self, config: SolverConfig = None):
        super().__init__()
        self.config = config or SolverConfig()
    
    def solve(self,
              topography: Topography,
              wind: WindField,
              atmosphere: AtmosphereState) -> FourierSolution:
        """
        Solve Fourier transform for given topography and wind.
        
        Args:
            topography: Topography grid
            wind: Wind field parameters
            atmosphere: Atmospheric state
            
        Returns:
            FourierSolution with coefficients and wavenumbers
        """
        # This would contain the actual FFT implementation
        # For now, return a placeholder
        
        grid = topography.grid
        n_s = grid.nx * 2  # With padding
        n_t = grid.ny
        
        h_hat = np.fft.fft2(topography.elevation, s=(n_s, n_t))
        k_z = np.ones((n_s, n_t), dtype=complex) * 0.001
        k_s = np.fft.fftfreq(n_s, d=grid.dx) * 2 * np.pi
        k_t = np.fft.fftfreq(n_t, d=grid.dy) * 2 * np.pi
        
        return FourierSolution(
            h_hat=h_hat,
            k_z=k_z,
            k_s=k_s,
            k_t=k_t,
            wind_coords=(grid.x, grid.y)
        )


class WindGridTransformer:
    """
    Transforms between geographic and wind-aligned coordinates.
    """
    
    @staticmethod
    def to_wind_coords(x: NDArray[np.float64],
                       y: NDArray[np.float64],
                       azimuth: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Transform geographic to wind-aligned coordinates.
        
        Args:
            x, y: Geographic coordinates
            azimuth: Wind direction in degrees
            
        Returns:
            s, t: Wind-aligned coordinates (s=downwind, t=crosswind)
        """
        az_rad = np.radians(azimuth)
        s = x * np.sin(az_rad) + y * np.cos(az_rad)
        t = -x * np.cos(az_rad) + y * np.sin(az_rad)
        return s, t
