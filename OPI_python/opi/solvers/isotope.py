"""
Isotope calculation using Rayleigh distillation model.
"""

import numpy as np
from numpy.typing import NDArray

from ..core.base import OPIBase
from ..models.results import IsotopeGrid, PrecipitationGrid
from ..models.parameters import IsotopeParameters, AtmosphereState


class IsotopeCalculator(OPIBase):
    """
    Calculates isotope composition using Rayleigh distillation.
    """
    
    def calculate(self,
                  precipitation: PrecipitationGrid,
                  isotope_params: IsotopeParameters,
                  atmosphere: AtmosphereState) -> IsotopeGrid:
        """
        Calculate isotope grid.
        
        Args:
            precipitation: Precipitation grid
            isotope_params: Isotope parameters
            atmosphere: Atmospheric state
            
        Returns:
            Isotope composition grid
        """
        # Placeholder: linear relationship with base value
        d2h_data = np.full_like(precipitation.data, isotope_params.base_d2h)
        
        return IsotopeGrid(data=d2h_data, grid=precipitation.grid)


class FractionationCalculator:
    """
    Calculates equilibrium and kinetic fractionation factors.
    """
    
    @staticmethod
    def equilibrium_alpha_hydrogen(temperature: float) -> float:
        """
        Calculate equilibrium fractionation factor for hydrogen.
        
        Uses Majoube (1971) relation.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Fractionation factor alpha
        """
        # ln(alpha) = A/T^2 + B/T^3 + ...
        # Simplified placeholder
        return np.exp(52.6 / temperature - 0.015)
    
    @staticmethod
    def equilibrium_alpha_oxygen(temperature: float) -> float:
        """
        Calculate equilibrium fractionation factor for oxygen.
        
        Uses Majoube (1971) relation.
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Fractionation factor alpha
        """
        # Simplified placeholder
        return np.exp(2.66 / temperature - 0.002)
    
    @staticmethod
    def kinetic_alpha(diffusivity_ratio: float, humidity: float) -> float:
        """
        Calculate kinetic fractionation factor.
        
        Args:
            diffusivity_ratio: Ratio of diffusivities
            humidity: Relative humidity (0-1)
            
        Returns:
            Kinetic fractionation factor
        """
        return 1 + (diffusivity_ratio - 1) * (1 - humidity)
