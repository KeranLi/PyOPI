"""
Precipitation calculation using LTOP model.
"""

import numpy as np
from numpy.typing import NDArray

from ..core.base import OPIBase
from ..models.results import PrecipitationGrid, FourierSolution
from ..models.parameters import AtmosphereState, MicrophysicsParameters
from ..models.domain import Grid


class PrecipitationCalculator(OPIBase):
    """
    Calculates precipitation rate using Linear Theory of Orographic Precipitation (LTOP).
    """
    
    def calculate(self,
                  fourier_solution: FourierSolution,
                  atmosphere: AtmosphereState,
                  microphysics: MicrophysicsParameters) -> PrecipitationGrid:
        """
        Calculate precipitation grid.
        
        Args:
            fourier_solution: Fourier solution from terrain
            atmosphere: Atmospheric state
            microphysics: Microphysical parameters
            
        Returns:
            Precipitation rate grid
        """
        # Placeholder implementation
        # Real implementation would use LTOP algorithm
        
        n_s, n_t = fourier_solution.h_hat.shape
        
        # Simple placeholder: precipitation proportional to terrain
        precip_data = np.abs(fourier_solution.h_hat[:n_s//2, :]) * 1e-6
        
        # Create grid (simplified)
        x = np.linspace(0, 100000, precip_data.shape[1])
        y = np.linspace(0, 100000, precip_data.shape[0])
        
        from ..models.domain import Grid
        grid = Grid(x=x, y=y)
        
        return PrecipitationGrid(data=precip_data, grid=grid)


class MoistureCalculator:
    """Calculates moisture-related quantities."""
    
    @staticmethod
    def calculate_vapor_scale_height(temperature: float) -> float:
        """
        Calculate water vapor scale height.
        
        H_s = R_d * T / g
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Scale height in meters
        """
        R_d = 287.0  # J/(kg·K)
        g = 9.81  # m/s²
        return R_d * temperature / g
